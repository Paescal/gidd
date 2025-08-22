import torch
from gidd.eval.visualize import history_to_str, marginals_to_str

class GenerationInfoHandler:
    def __init__(self, max_seq_len, info:list=[]):
        self.num_samples = 0
        self.max_seq_len = max_seq_len
        if "history" in info:
            self.collect_history = True
            self.history = None
        if "marginals" in info:
            self.collect_marginals = True
            self.denoised_fraction = None
            self.mask_fraction = None
            self.uniform_fraction = None

    def batch_initialize(self, initial_z_t, diffusion_mask, solution, device):
        self.batch_size = initial_z_t.shape[0]
        self.diffusion_mask = diffusion_mask.to(device=device, non_blocking=True)
        self.num_diffusion_positions = diffusion_mask.sum()
        self.batch_solution = solution.to(device=device, non_blocking=True)
        if self.collect_history:
            self.batch_history = [initial_z_t.clone().to(device=device, non_blocking=True)]
        if self.collect_marginals:
            self.batch_denoised_fraction = [0.]
            self.batch_mask_fraction = [self.batch_size]
            self.batch_uniform_fraction = [0.]
    
    def batch_step_history(self, z_t):
        if self.collect_history:
            self.batch_history.append(z_t.clone())

    def batch_step_marginals(self, z_t, p_zt_x, mask_token_id):
        if self.collect_marginals:
            current_denoised_fraction = ((p_zt_x * self.diffusion_mask).sum() / self.num_diffusion_positions).item()
            current_mask_fraction = (((z_t == mask_token_id) * self.diffusion_mask).sum() / self.num_diffusion_positions).item()
            self.batch_denoised_fraction.append(current_denoised_fraction * self.batch_size)
            self.batch_mask_fraction.append(current_mask_fraction * self.batch_size)
            self.batch_uniform_fraction.append((1 - current_denoised_fraction - current_mask_fraction) * self.batch_size)

    def batch_finalize(self):
        self.num_samples += self.batch_size
        if self.collect_history:
            self.batch_history = torch.stack(self.batch_history, dim=0).permute(1, 0, 2)
            self.batch_history = torch.cat([self.batch_history, self.batch_solution.unsqueeze(1)], dim=1)
            if self.history is None:
                self.history = self.batch_history
            else:
                if self.history.shape[1] != self.batch_history.shape[1] or self.history.shape[2] != self.batch_history.shape[2]:
                    raise ValueError("Inconsistent history shapes")
                self.history = torch.cat([self.history, self.batch_history], dim=0)
        if self.collect_marginals:
            self.batch_denoised_fraction = torch.tensor(self.batch_denoised_fraction)
            self.batch_mask_fraction = torch.tensor(self.batch_mask_fraction)
            self.batch_uniform_fraction = torch.tensor(self.batch_uniform_fraction)
            if self.denoised_fraction is None:
                self.denoised_fraction = self.batch_denoised_fraction
                self.mask_fraction = self.batch_mask_fraction
                self.uniform_fraction = self.batch_uniform_fraction
            else:
                if self.denoised_fraction.shape != self.batch_denoised_fraction.shape:
                    raise ValueError("Inconsistent marginals shapes")
                self.denoised_fraction = self.denoised_fraction + self.batch_denoised_fraction
                self.mask_fraction = self.mask_fraction + self.batch_mask_fraction
                self.uniform_fraction = self.uniform_fraction + self.batch_uniform_fraction
    
    def get_history(self):
        if self.collect_history:
            return self.history
        else:
            raise ValueError("History was not collected")
    
    def get_marginals(self):
        if self.collect_marginals:
            return {
                "denoised_fraction": self.denoised_fraction / self.num_samples,
                "mask_fraction": self.mask_fraction / self.num_samples,
                "uniform_fraction": self.uniform_fraction / self.num_samples
            }
        else:
            raise ValueError("Marginals were not collected")
    
    def print_info(self):
        if self.collect_history:
            chosen_sample_for_history = 19
            history = self.get_history().cpu()
            history_of_chosen_sample = history[chosen_sample_for_history, :, -self.max_seq_len:]
            print(history_to_str(history_of_chosen_sample[:-1], history_of_chosen_sample[-1]))
        if self.collect_marginals:
            marginals = self.get_marginals()
            print(marginals_to_str(marginals["denoised_fraction"].cpu(), marginals["mask_fraction"].cpu(), marginals["uniform_fraction"].cpu()))