import random
import torch


# modified from AnoDDPM repo to enable batched simplex noise generation
def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
        ):
    noise = torch.empty(x.shape).to(x.device)
    
    for n in range(x.shape[0]):
            
        for i in range(in_channels):
                Simplex_instance.newSeed()
                if random_param:
                        param = random.choice(
                                [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                                (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                                (2, 0.85, 8),
                                (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                                (1, 0.85, 8),
                                (1, 0.85, 4), (1, 0.85, 2), ]
                                )
                        # 2D octaves seem to introduce directional artifacts in the top left
                        noise[n, i, ...] = torch.unsqueeze(
                                torch.from_numpy(
                                        # Simplex_instance.rand_2d_octaves(
                                        #         x.shape[-2:], param[0], param[1],
                                        #         param[2]
                                        #         )
                                        Simplex_instance.rand_3d_fixed_T_octaves(
                                                x.shape[-2:], t.detach().cpu().numpy()[n:n+1], param[0], param[1],
                                                param[2]
                                                )
                                        ).to(x.device), 0
                                ).repeat(1, 1, 1, 1)
                        
                noise[n, i, ...] = torch.unsqueeze(
                        torch.from_numpy(
                                # Simplex_instance.rand_2d_octaves(
                                #         x.shape[-2:], octave,
                                #         persistence, frequency
                                #         )
                                Simplex_instance.rand_3d_fixed_T_octaves(
                                        x.shape[-2:], t.detach().cpu().numpy()[n:n+1], octave,
                                        persistence, frequency
                                        )
                                ).to(x.device), 0
                        ).repeat(1, 1, 1, 1)
    return noise

# TODO: implement the following function
def generate_coarse_noise():
        return None