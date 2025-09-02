import torch
import time
from typing import Optional, Dict, Any
from gidd.eval.visualize import history_to_str, marginals_to_str
# from gidd.eval.evaluate_one import namespace_to_dict

class GenerationInfoHandler:
    def __init__(self, max_seq_len, sampling_config_dict, noise_schedule, info:list=[]):
        self.num_samples = 0
        self.max_seq_len = max_seq_len
        self.sampling_config_dict = sampling_config_dict
        self.noise_schedule = noise_schedule
        if "history" in info:
            self.collect_history = True
            self.history = None
        if "marginals" in info:
            self.collect_marginals = True
            self.true_denoised_fraction = None # averaged over samples
            self.denoised_fraction = None # averaged over samples
            self.mask_fraction = None # averaged over samples
            self.uniform_fraction = None # averaged over samples
            self.p_zt_x_mean = None # taken over positions, per sample
            self.p_zt_x_std = None # taken over positions, per sample
            self.p_zt_x_min = None # taken over positions, per sample
            self.p_zt_x_max = None # taken over positions, per sample
            self.cell_accuracy = None # average over positions, per sample
            self.expected_cell_accuracy = None # values over time steps (same for all positions and samples)
            self.total_accuracy = None # averaged over samples
        if "change_events" in info:
            self.collect_change_events = True
            self.change_events = None # list of lists of dicts

    def batch_initialize(self, initial_z_t, diffusion_mask, solution, device):
        self.batch_size = initial_z_t.shape[0]
        self.batch_seq_len = initial_z_t.shape[1]
        self.diffusion_mask = diffusion_mask.to(device=device, non_blocking=True)
        self.num_diffusion_positions_by_sample = self.diffusion_mask.sum(dim=1)
        self.batch_solution = solution.to(device=device, non_blocking=True)
        if getattr(self, "collect_history", False):
            self.batch_history = [initial_z_t.clone().to(device=device, non_blocking=True)[:, -self.max_seq_len:]]
        if getattr(self, "collect_marginals", False):
            self.batch_true_denoised_fraction = []
            self.batch_denoised_fraction = []
            self.batch_mask_fraction = []
            self.batch_uniform_fraction = []
            self.batch_p_zt_x_mean = []
            self.batch_p_zt_x_std = []
            self.batch_p_zt_x_min = []
            self.batch_p_zt_x_max = []
            self.batch_cell_accuracy = []
            self.batch_expected_cell_accuracy = []
            self.batch_total_accuracy = []
        if getattr(self, "collect_change_events", False):
            self.old_z_t = initial_z_t.clone().to(device=device, non_blocking=True)
            self.batch_change_events = [[] for _ in range(self.batch_size)]
            
    
    def batch_step_history(self, z_t):
        if getattr(self, "collect_history", False):
            self.batch_history.append(z_t.clone()[:, -self.max_seq_len:])

    def batch_step_marginals(self, z_t, p_zt_x, t, mask_token_id):
        if getattr(self, "collect_marginals", False):
            cell_accuracy_by_sample = ((z_t == self.batch_solution) * self.diffusion_mask).sum(dim=-1) / self.num_diffusion_positions_by_sample
            total_accuracy_by_sample = ((z_t == self.batch_solution) | ~self.diffusion_mask.bool()).all(dim=-1)
            mask_fraction_by_sample = ((z_t == mask_token_id) * self.diffusion_mask).sum(dim=-1) / self.num_diffusion_positions_by_sample
            p_zt_x_mean_by_sample = (p_zt_x * self.diffusion_mask).sum(dim=-1) / self.num_diffusion_positions_by_sample
            p_zt_x_std_by_sample = torch.sqrt((torch.pow(p_zt_x - p_zt_x_mean_by_sample.unsqueeze(-1), 2) * self.diffusion_mask).sum(dim=-1) / (self.num_diffusion_positions_by_sample - 1).clamp(min=0)) # nan if less than 2 diffusion positions
            p_zt_x_min_by_sample = torch.where(self.diffusion_mask.bool(), p_zt_x, 1).min(dim=-1).values
            p_zt_x_max_by_sample = torch.where(self.diffusion_mask.bool(), p_zt_x, 0).max(dim=-1).values
            alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            expected_cell_accuracy = alpha_t[0, 0].item() + beta_pi_t[0, self.noise_schedule.not_mask_id].item()

            self.batch_true_denoised_fraction.append(cell_accuracy_by_sample.sum().item())
            self.batch_denoised_fraction.append(p_zt_x_mean_by_sample.sum().item())
            self.batch_mask_fraction.append(mask_fraction_by_sample.sum().item())
            self.batch_uniform_fraction.append((1 - p_zt_x_mean_by_sample - mask_fraction_by_sample).sum().item())
            self.batch_p_zt_x_mean.append(p_zt_x_mean_by_sample)
            self.batch_p_zt_x_std.append(p_zt_x_std_by_sample)
            self.batch_p_zt_x_min.append(p_zt_x_min_by_sample)
            self.batch_p_zt_x_max.append(p_zt_x_max_by_sample)
            self.batch_cell_accuracy.append(cell_accuracy_by_sample)
            self.batch_expected_cell_accuracy.append(expected_cell_accuracy)
            self.batch_total_accuracy.append(total_accuracy_by_sample.sum().item())

    def batch_step_change_events(self, step, new_z_t, model_confidence_new_z_t, mask_token_id):
        def get_change_event_type(old_value, new_value, true_value, mask_token_id):
            old_is_uniform_token = old_value != mask_token_id and old_value != true_value
            new_is_uniform_token = new_value != mask_token_id and new_value != true_value
            if old_value == mask_token_id and new_value == true_value:
                return "mask_to_denoised"
            elif old_value == mask_token_id and new_is_uniform_token:
                return "mask_to_uniform"
            elif old_is_uniform_token and new_value == true_value:
                return "uniform_to_denoised"
            elif old_is_uniform_token and new_is_uniform_token:
                return "uniform_to_uniform"
            elif old_value == true_value and new_is_uniform_token:
                return "denoised_to_uniform"
            else:
                raise ValueError(f"Unexpected change from {old_value} to {new_value} (true value: {true_value})")

        if getattr(self, "collect_change_events", False):
            new_z_t = new_z_t.clone().detach()
            has_changed = (new_z_t != self.old_z_t)
            change_indices_rows, change_indices_columns = has_changed.nonzero(as_tuple=True)
            old_values = self.old_z_t[change_indices_rows, change_indices_columns]
            new_values = new_z_t[change_indices_rows, change_indices_columns]

            for sample, position, old_value, new_value in zip(change_indices_rows.tolist(), change_indices_columns.tolist(), old_values.cpu().tolist(), new_values.cpu().tolist()):
                self.batch_change_events[sample].append({
                    "step": step,
                    "position": position - self.batch_seq_len + self.max_seq_len,
                    "old_value": old_value,
                    "new_value": new_value,
                    "event_type": get_change_event_type(old_value, new_value, self.batch_solution[sample, position].item(), mask_token_id),
                    "model_confidence": model_confidence_new_z_t[sample, position].item(),
                })

            self.old_z_t = new_z_t

    def batch_finalize(self):
        self.num_samples += self.batch_size
        if getattr(self, "collect_history", False):
            self.batch_history = torch.stack(self.batch_history, dim=1)
            self.batch_history = torch.cat([self.batch_history, self.batch_solution[:, -self.max_seq_len:].unsqueeze(1)], dim=1)
            if self.history is None:
                self.history = self.batch_history
            else:
                if self.history.shape[1] != self.batch_history.shape[1] or self.history.shape[2] != self.batch_history.shape[2]:
                    raise ValueError("Inconsistent history shapes")
                self.history = torch.cat([self.history, self.batch_history], dim=0)
        if getattr(self, "collect_marginals", False):
            self.batch_true_denoised_fraction = torch.tensor(self.batch_true_denoised_fraction)
            self.batch_denoised_fraction = torch.tensor(self.batch_denoised_fraction)
            self.batch_mask_fraction = torch.tensor(self.batch_mask_fraction)
            self.batch_uniform_fraction = torch.tensor(self.batch_uniform_fraction)
            self.batch_total_accuracy = torch.tensor(self.batch_total_accuracy)
            if self.true_denoised_fraction is None:
                self.true_denoised_fraction = self.batch_true_denoised_fraction
                self.denoised_fraction = self.batch_denoised_fraction
                self.mask_fraction = self.batch_mask_fraction
                self.uniform_fraction = self.batch_uniform_fraction
                self.p_zt_x_mean = torch.stack(self.batch_p_zt_x_mean, dim=-1)
                self.p_zt_x_std = torch.stack(self.batch_p_zt_x_std, dim=-1)
                self.p_zt_x_min = torch.stack(self.batch_p_zt_x_min, dim=-1)
                self.p_zt_x_max = torch.stack(self.batch_p_zt_x_max, dim=-1)
                self.cell_accuracy = torch.stack(self.batch_cell_accuracy, dim=-1)
                self.expected_cell_accuracy = torch.tensor(self.batch_expected_cell_accuracy)
                self.total_accuracy = self.batch_total_accuracy
            else:
                if self.true_denoised_fraction.shape != self.batch_true_denoised_fraction.shape:
                    raise ValueError("Inconsistent marginals shapes")
                self.true_denoised_fraction = self.true_denoised_fraction + self.batch_true_denoised_fraction
                self.denoised_fraction = self.denoised_fraction + self.batch_denoised_fraction
                self.mask_fraction = self.mask_fraction + self.batch_mask_fraction
                self.uniform_fraction = self.uniform_fraction + self.batch_uniform_fraction
                self.p_zt_x_mean = torch.cat([self.p_zt_x_mean, torch.stack(self.batch_p_zt_x_mean, dim=-1)], dim=0)
                self.p_zt_x_std = torch.cat([self.p_zt_x_std, torch.stack(self.batch_p_zt_x_std, dim=-1)], dim=0)
                self.p_zt_x_min = torch.cat([self.p_zt_x_min, torch.stack(self.batch_p_zt_x_min, dim=-1)], dim=0)
                self.p_zt_x_max = torch.cat([self.p_zt_x_max, torch.stack(self.batch_p_zt_x_max, dim=-1)], dim=0)
                self.cell_accuracy = torch.cat([self.cell_accuracy, torch.stack(self.batch_cell_accuracy, dim=-1)], dim=0)
                self.expected_cell_accuracy = self.expected_cell_accuracy # NOP, as long as the number of denoising steps stays the same, so does this
                self.total_accuracy = self.total_accuracy + self.batch_total_accuracy
        if getattr(self, "collect_change_events", False):
            if self.change_events is None:
                self.change_events = self.batch_change_events
            else:
                self.change_events.extend(self.batch_change_events)

    def get_history(self):
        if getattr(self, "collect_history", False):
            return self.history # shape (num_samples, num_denoising_steps, seq_len)
        else:
            raise ValueError("History was not collected")
    
    def get_marginals(self):
        if getattr(self, "collect_marginals", False):
            return {
                "true_denoised_fraction": self.true_denoised_fraction / self.num_samples, # shape (num_denoising_steps)
                "denoised_fraction": self.denoised_fraction / self.num_samples, # shape (num_denoising_steps)
                "mask_fraction": self.mask_fraction / self.num_samples, # shape (num_denoising_steps)
                "uniform_fraction": self.uniform_fraction / self.num_samples, # shape (num_denoising_steps)
                "p_zt_x_mean": self.p_zt_x_mean, # shape (num_samples, num_denoising_steps)
                "p_zt_x_std": self.p_zt_x_std, # shape (num_samples, num_denoising_steps)
                "p_zt_x_min": self.p_zt_x_min, # shape (num_samples, num_denoising_steps)
                "p_zt_x_max": self.p_zt_x_max, # shape (num_samples, num_denoising_steps)
                "cell_accuracy": self.cell_accuracy, # shape (num_samples, num_denoising_steps)
                "expected_cell_accuracy": self.expected_cell_accuracy, # shape (num_denoising_steps)
                "total_accuracy": self.total_accuracy / self.num_samples, # shape (num_denoising_steps)
            }
        else:
            raise ValueError("Marginals were not collected")
    
    def get_change_events(self):
        if getattr(self, "collect_change_events", False):
            return self.change_events # list of num_samples lists, each containing num_changes change events, each change event is a dict
        else:
            raise ValueError("Change events were not collected")

    def print_info(self):
        if getattr(self, "collect_history", False):
            chosen_sample_for_history = 19
            history = self.get_history().cpu()
            history_of_chosen_sample = history[chosen_sample_for_history, :, -self.max_seq_len:]
            print(history_to_str(history_of_chosen_sample[:-1], history_of_chosen_sample[-1]))
        if getattr(self, "collect_marginals", False):
            marginals = self.get_marginals()
            print(marginals_to_str(marginals["true_denoised_fraction"].cpu(), marginals["denoised_fraction"].cpu(), marginals["mask_fraction"].cpu(), marginals["uniform_fraction"].cpu()))
    
    def save_info(self, out_dir: str, meta: Optional[Dict[str, Any]] = None):
        """
        Saves collected data to `out_dir` with a stable schema.
        """
        from io_generation_info import (
            _ensure_dir, save_meta, save_history, save_marginals, save_change_events_table
        )
        _ensure_dir(out_dir)

        meta = dict(meta or {})
        meta.update({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.gmtime()),
            "num_samples": getattr(self, "num_samples", None),
            "max_seq_len": getattr(self, "max_seq_len", None),
            "collect_history": getattr(self, "collect_history", False),
            "collect_marginals": getattr(self, "collect_marginals", False),
            "collect_change_events": getattr(self, "collect_change_events", False),
        })
        meta.update(self.sampling_config_dict)

        if getattr(self, "collect_history", False):
            history = self.get_history().detach().cpu()
            meta["history_shape"] = list(history.shape)
            save_history(history, out_dir)

        if getattr(self, "collect_marginals", False):
            marginals = self.get_marginals()
            # ensure CPU for portability
            marginals_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in marginals.items()}
            # capture shapes
            meta["marginals_shapes"] = {k: (list(v.shape) if torch.is_tensor(v) else None) for k, v in marginals_cpu.items()}
            save_marginals(marginals_cpu, out_dir)

        if getattr(self, "collect_change_events", False):
            save_change_events_table(self.get_change_events(), out_dir)

        save_meta(meta, out_dir)

    @staticmethod
    def load_all(out_dir: str, map_location: str = "cpu"):
        """
        Convenience loader that returns (meta, history, marginals, change_events_df_or_list)
        """
        from io_generation_info import load_meta, load_history, load_marginals, load_change_events_table
        meta = load_meta(out_dir)
        history = load_history(out_dir, map_location) if meta.get("collect_history") else None
        marginals = load_marginals(out_dir, map_location) if meta.get("collect_marginals") else None
        # may return pandas.DataFrame or list of dicts depending on availability
        change_events_table = load_change_events_table(out_dir) if meta.get("collect_change_events") else None
        return meta, history, marginals, change_events_table