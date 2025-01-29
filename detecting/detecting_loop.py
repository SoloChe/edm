import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from detecting.detecting_func import VPPerturbation
from detecting.detecting_func import post_reconstruction

def detecting_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_pkl         = '.',      # Options for model and preconditioning.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    total_kimg          = 1000,     # Training duration, measured in thousands of training images.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    steps_noise         = 100,      # noise level of latent space
    steps_denoise       = 100,      # number of steps for denoising
    sampler_fn          = None,     # function to sample from the latent space
    recon_kwargs        = {},       # Custom op implementations.
    postrecon_kwargs    = {},       # Custom op implementations.
    custom_kwargs       = {},       # Custom op implementations.
):
    
     # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
        
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
        
     # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
        
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
        
    # Noise setup
    if custom_kwargs.gaussian:
        noise_fn = lambda x, t=None: torch.randn_like(x)
    else:
        from noise.noise import generate_simplex_noise
        from noise.simplex import Simplex_CLASS
        simplex = Simplex_CLASS()
        noise_fn = lambda x, t: generate_simplex_noise(
                                                        simplex,
                                                        x,
                                                        t,
                                                        False,
                                                        in_channels=1,
                                                        octave=6,
                                                        persistence=0.8,
                                                        frequency=64,
                                                        )
    recon_kwargs.randn_like = noise_fn
    
    # Loop over testing images
    dist.print0(f'Testing {total_kimg} k images...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    perturbator = VPPerturbation()
    
    while True:
        
        data, labels = next(dataset_iterator)
        images, masks, C = data
        # original divided by 99th percentile and clip to [0, 1]
        images = images.to(device).to(torch.float32) * 2 - 1
        masks = masks.to(device).to(torch.float32)
        if custom_kwargs.grad:
            C = C.to(device).to(torch.float32)
            C = (C - C.max()).abs() # low frequency component has higher value
            C = C / 5.27 # 98th percentile (max)
            custom_kwargs.C = C
        
        # detection/segmentation
        # TODO: implement the following function
        recon_images, latents = perturbator.reconstruction(net, images, steps_noise, steps_denoise, sampler_fn,
                                                           custom_kwargs, **recon_kwargs)
        # all components gathered at rank 0 and return at rank 0
        metrics = post_reconstruction(run_dir, cur_tick, 
                                      images=images, 
                                      latents=latents,
                                      recon_images=recon_images, 
                                      masks=masks,
                                      **postrecon_kwargs
                                      )
                
        cur_nimg += batch_gpu*dist.get_world_size()
        done = (cur_nimg >= total_kimg * 1000)
        
        training_stats.report0('DICE', metrics['dice'])
        training_stats.report0('IOU', metrics['iou'])
        training_stats.report0('AUPRC', metrics['AUPRC'])
        
        # Print status line, accumulating the same information in training_stats.
        # TODO: Modify the following code to print the metrics
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
            
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
        
        # Update logs.
        training_stats.default_collector.update()
        
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        
        if done:
            break
        
        
    # Done.
    dist.print0()
    dist.print0('Exiting...')
        

    