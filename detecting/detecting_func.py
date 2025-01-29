import torch
import torch.distributed as dist
import os
import numpy as np
from detecting.metrics import evaluate

class VPPerturbation:
    def __init__(self):
        self.beta_d = 20-0.1
        self.beta_min = 0.1
        self.M = 1000
        self.epsilon_s = 1e-3
        
        self.mean = lambda t: torch.exp(-1/4*t**2 * self.beta_d - 1/2*t * self.beta_min)
        self.std = lambda t: torch.sqrt( 1 - torch.exp(-1/2*t**2 * self.beta_d - t * self.beta_min) )
        self.original_t_M = lambda step_idx: 1 + (self.M - step_idx) / (self.M - 1) * (self.epsilon_s - 1) # [0, M-1] -> [epsilon_s, 1]; original: [0, M-1] -> [1, epsilon_s]
    
    def adding_noise(self, images,  steps, noise_fn, **kwargs):
        
        t = self.original_t_M(steps) # steps -> t
        t = t*torch.ones(images.shape[0], 1, 1, 1).to(images.device)
             
        if 'C' in kwargs:
            C = kwargs['a']*kwargs['C'] + kwargs['b']
            C = C.sqrt()
            sigma = sigma * C.sqrt()
            
        latents = noise_fn(images, steps) * self.std(t) + self.mean(t)*images
        return latents

    def reconstruction(self, net, images, steps_noise, steps_denoise, sampler_fn, noise_kwargs, **kwargs):
        class_labels = None
        latents = self.adding_noise(images, steps_noise, kwargs['randn_like'], **noise_kwargs)
        recon_images = sampler_fn(net, latents, class_labels,  
                                  steps_noise=steps_noise, 
                                  steps_denoise=steps_denoise, 
                                  **kwargs)
        return recon_images, latents

def post_reconstruction(run_dir, batch_idx, images, latents, recon_images, masks, **kwargs):
    # collect
    dict_triplet_np = {}
    
    collected_images = [torch.zeros_like(images) for _ in range(dist.get_world_size()) if dist.get_rank() == 0]
    dist.all_gather(collected_images, images)
    
    collected_recon_images = [torch.zeros_like(recon_images) for _ in range(dist.get_world_size()) if dist.get_rank() == 0]
    dist.all_gather(collected_recon_images, recon_images)
    
    collected_latents = [torch.zeros_like(latents) for _ in range(dist.get_world_size()) if dist.get_rank() == 0]
    dist.all_gather(collected_latents, latents)
    
    collected_masks = [torch.zeros_like(masks) for _ in range(dist.get_world_size()) if dist.get_rank() == 0]
    dist.all_gather(collected_masks, masks)
    
    if dist.get_rank() == 0:
        dict_triplet_np['images'] = torch.cat(collected_images, dim=0).cpu().numpy()
        dict_triplet_np['recon_images'] = torch.cat(collected_recon_images, dim=0).cpu().numpy()
        dict_triplet_np['latents'] = torch.cat(collected_latents, dim=0).cpu().numpy()
        dict_triplet_np['masks'] = torch.cat(collected_masks, dim=0).cpu().numpy().astype(np.int8)
    
        metrics, ano_maps, recon_masks = evaluate(dict_triplet_np['images'], 
                                                  dict_triplet_np['recon_images'], 
                                                  dict_triplet_np['masks'],
                                                  threshold=kwargs['threshold'],
                                                  cc_filter=kwargs['cc_filter'],
                                                  mixed=kwargs['mixed'])
        dict_triplet_np['ano_maps'] = ano_maps
        dict_triplet_np['recon_masks'] = recon_masks
        np.savez(os.path.join(run_dir, f'{batch_idx}'), **dict_triplet_np)
        return metrics
        
        
        
        
    