from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from gidd.diffusion_process import NoiseSchedule
from gidd.utils import get_position_metric, get_position_sampling_strategy, get_token_sampling_strategy, sample_categorical
from gidd.sampling_strategies import get_sampling_strategy

class Sampler(nn.Module):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps: float = 1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps

    @abstractmethod
    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, num_samples=1, num_denoising_steps=1000, max_length=None, decode=True, show_progress=True):
        max_length = max_length or self.model.config.max_seq_len
        device = next(self.model.parameters()).device

        z_t = self._do_generate(num_samples, num_denoising_steps, max_length, show_progress=show_progress, device=device)

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            return z_t
    
    @abstractmethod
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, num_denoising_steps, max_length, show_progress, device, keep_history=False):
        raise NotImplementedError
    
    @abstractmethod
    def _do_generate_from_expected_t(self, initial_z_t, diffusion_mask, num_denoising_steps, add_random_tokens, max_length, show_progress, device, keep_history=False):
        raise NotImplementedError
    
    # Generate starting from a given z_t, the diffusion_mask is a tensor with the same shape as z_t of 0s and 1s, where 1 indicates that the token is NOT given and needs to be denoised
    @torch.no_grad()
    def generate_from_given(self, z_t, diffusion_mask, puzzle_conditioning=None, num_denoising_steps=128, max_length=None, decode=True, show_progress=True, keep_history=False):
        max_length = max_length or self.model.config.model.max_seq_len
        # print("getting device")
        device = next(self.model.parameters()).device

        # print("calling _do_generate_from_given")
        if puzzle_conditioning is None:
            z_t, history = self._do_generate_from_given(z_t, diffusion_mask=diffusion_mask, puzzle_conditioning=z_t.clone(), num_denoising_steps=num_denoising_steps, max_length=max_length, show_progress=show_progress, device=device, keep_history=keep_history)
        else:
            z_t, history = self._do_generate_from_given(z_t, diffusion_mask=diffusion_mask, puzzle_conditioning=puzzle_conditioning, num_denoising_steps=num_denoising_steps, max_length=max_length, show_progress=show_progress, device=device, keep_history=keep_history)
        # print("done calling _do_generate_from_given")

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            if keep_history:
                return z_t, history
            else:
                return z_t
    @torch.no_grad()
    def generate_from_expected_t(self, z_t, diffusion_mask, num_denoising_steps=128, add_random_tokens=True, max_length=None, decode=True, show_progress=True, keep_history=False):
        max_length = max_length or self.model.config.max_seq_len
        # print("getting device")
        device = next(self.model.parameters()).device

        # print("calling _do_generate_from_expected_t")
        z_t, history = self._do_generate_from_expected_t(z_t, diffusion_mask=diffusion_mask, num_denoising_steps=num_denoising_steps, add_random_tokens=add_random_tokens, max_length=max_length, show_progress=show_progress, device=device, keep_history=keep_history)
        # print("done calling _do_generate_from_expected_t")

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            if keep_history:
                return z_t, history
            else:
                return z_t

class GiddSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, config, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.min_p = min_p
            # self.position_metric = get_position_metric(config, tokenizer)
            # self.position_sampling_strategy = get_position_sampling_strategy(config)
            # self.token_sampling_strategy = get_token_sampling_strategy(config, tokenizer)
            self.sampling_strategy = get_sampling_strategy(config, tokenizer, noise_schedule=noise_schedule, min_p=min_p)
            self.config = config

        def forward(self, z_t, t, s, diffusion_mask=None, puzzle_conditioning=None, sequence_length_without_conditioning=0):
            is_fully_denoised = (z_t[:, -sequence_length_without_conditioning:] != self.tokenizer.mask_token_id).all(dim=1)
            # if is_fully_denoised.sum() != 0:
            #     print(f"is_fully_denoised: {is_fully_denoised.sum()}")
            # print("inside gidd denoising step")
            logits = self.model(z_t, t, puzzle_conditioning=puzzle_conditioning)
            # print("got logits from model")
            # print(f"logits: {logits[..., self.tokenizer.mask_token_id - 1]}")
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)

            # if self.config.sampling.position_sampling_strategy == "all" or self.config.sampling.position_sampling_strategy == "independent" or self.config.sampling.position_metric == "probs_to_change":
            #     # if i > 0:
            #     # print("getting probs at t and s")
            #     q_s = self.noise_schedule.probs_at_t(probs, s)
            #     q_t = self.noise_schedule.probs_at_t(probs, t)
            #     q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            #     # print("getting alpha and beta pi at t and s")
            #     alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            #     alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

            #     alpha_ts = alpha_t / alpha_s
            #     beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            #     # vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
            #     vocab_size_architecturally = len(self.tokenizer)
            #     vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
            #     beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            #     q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

            #     q_st = q_ts * q_s / q_zt
            #     # print(f"shapes: q_st: {q_st.shape}, q_ts: {q_ts.shape}, q_s: {q_s.shape}, q_zt: {q_zt.shape}, beta_pi_ts_at_zt: {beta_pi_ts_at_zt.shape}, alpha_ts: {alpha_ts.shape}, t: {t.shape}")
            #     # print(f"q_st: {q_st[0, -2, self.tokenizer.mask_token_id]}, q_ts: {q_ts[0, -2, self.tokenizer.mask_token_id]}, q_s: {q_s[0, -2, self.tokenizer.mask_token_id]}, q_zt: {q_zt[0, -2, 0]}, beta_pi_ts_at_zt: {beta_pi_ts_at_zt[0, -2, 0]}")
                
            #     if self.min_p > 0.0:
            #         is_small = (q_st < self.min_p).float()
            #         q_st = (1 - is_small) * q_st
            #         q_st = q_st / q_st.sum(-1, keepdim=True)
            #     if self.config.sampling.position_sampling_strategy == "all":
            #         probs = q_st
                    
            # if self.config.sampling.position_metric == "probs_to_change":
            #     q_st_at_zt = q_st.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
            #     # token_start = 81
            #     # token_end = 86
            #     # print(f"q_st: {q_st[0, token, :10]}")
            #     # print(f"q_st_at_zt: {q_st_at_zt[0, token:token+5]}\nz_t: {z_t[0, token:token+5]}")
                
            #     # Calculate the probability to unmask a token,
            #     # compare it with the probabilities to change a token.
            #     # If there exists a token which is more likely to change than it is likely to unmask a token, then change it.
            #     # Otherwise unmask a token based on the current criteria (position_metric, e.g. max confidence, etc.)
                
            #     # P(masked s) = P(masked s | unmasked t) * P(unmasked t) + P(masked s | masked t) * P(masked t), P(masked s | unmasked t) = 0
            #     # -> P(masked s | masked t) = P(masked s) / P(masked t)
            #     # -> P(unmasked s | masked t) = 1 - P(masked s) / P(masked t)

            #     # P(masked s | masked t) = P(masked t | masked s) * P(masked s) / P(masked t), P(masked t | masked s) = 1
            #     # -> P(masked s | masked t) = P(masked s) / P(masked t)
            #     probs_to_change = 1 - q_st_at_zt
            #     beta_pi_s_at_mask = beta_pi_s[..., self.tokenizer.mask_token_id]
            #     beta_pi_t_at_mask = beta_pi_t[..., self.tokenizer.mask_token_id]
            #     probs_to_not_unmask = beta_pi_s_at_mask / beta_pi_t_at_mask
            #     probs_to_not_unmask = probs_to_not_unmask * (alpha_t.squeeze(-1) / alpha_s.squeeze(-1) + beta_pi_t_at_mask - alpha_t.squeeze(-1) / alpha_s.squeeze(-1) * beta_pi_s_at_mask)
            #     probs_to_unmask = 1 - probs_to_not_unmask
            #     # print(f"Shape of probs to not unmask: {probs_to_not_unmask.shape}, beta_pi_s_at_mask: {beta_pi_s_at_mask.shape}")
            #     # print(f"probs to not unmask: {1 - probs_to_unmask[0]} ( = {beta_pi_s_at_mask.item()} / {beta_pi_t_at_mask.item()})")
            #     # print(f"probs to not change: {q_st_at_zt[0, token_start:token_end]}")
            #     # print(f"q_ts factor: {(alpha_t / alpha_s + beta_pi_t_at_mask - alpha_t / alpha_s * beta_pi_s_at_mask)[0, 0].item()}")
            #     probs_to_unmask = probs_to_unmask.unsqueeze(-1).expand_as(probs_to_change)

            #     probs_to_change = probs_to_change * diffusion_mask
            #     probs_to_unmask = probs_to_unmask * diffusion_mask

            #     positions_to_change = (probs_to_change > probs_to_unmask + 1e-6)
            #     # print(f"z_t: {z_t[0, token_start:token_end]}")
            #     # print(f"probs_to_change: {probs_to_change[0, token_start:token_end]}")
            #     # print(f"probs_to_unmask: {probs_to_unmask[0, token_start:token_end]}")

            #     if positions_to_change.any():
            #         # print("Changing tokens")
            #         # print(f"Positions to change: {positions_to_change[0, token_start:token_end]}")
            #         metric = probs_to_change * positions_to_change.to(dtype=probs_to_change.dtype)
            #     else:
            #         # print("Unmasking tokens")
            #         metric = self.position_metric(z_t, probs)
            # else:
            #     metric = self.position_metric(z_t, probs)
            # metric = metric * diffusion_mask

            # if self.config.sampling.position_sampling_strategy == "independent":
            #     update_positions = self.position_sampling_strategy(metric, (alpha_s - alpha_t) / (1 - alpha_t))
            # else:
            #     update_positions = self.position_sampling_strategy(metric)
            # update_positions = update_positions * diffusion_mask
            # # print(f"update_positions: {update_positions[0, token:token+5]}")
            
            # if self.config.sampling.token_sampling_strategy == "change_token_max":
            #     next_z_t = self.token_sampling_strategy(probs, z_t)
            # else:
            #     next_z_t = self.token_sampling_strategy(probs)

            update_positions, next_z_t = self.sampling_strategy(probs, z_t, t, s, diffusion_mask)
            update_positions = update_positions * diffusion_mask
            update_positions = update_positions & ~is_fully_denoised.unsqueeze(-1)
            return torch.where(update_positions.bool(), next_z_t, z_t)

    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(config, model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        # zt = sample_categorical(p_zt)
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        return z_t
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, puzzle_conditioning, num_denoising_steps, max_length, show_progress, device, keep_history=False):
        debug_flag = False

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps
        
        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        
        z_t = initial_z_t.clone()
        
        history = [initial_z_t.clone()] if keep_history or debug_flag else None
        
        # print("entering sampling loop in _do_generate_from_given")
        mask_token_id = self.sampling_step.tokenizer.mask_token_id
        # initial_num_mask_tokens = (z_t[:, -max_length:] == mask_token_id).sum()
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            # print(f"sampling step {i}")
            old_z_t = z_t.clone()
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=diffusion_mask, puzzle_conditioning=puzzle_conditioning, sequence_length_without_conditioning=max_length)
            # print(f"sampling step {i} done")
            
            puzzle_string = ''
            sample_in_batch = 1
            changes_mask = (old_z_t[sample_in_batch] != z_t[sample_in_batch]).to(dtype=int)
            indices_of_change = torch.tensor([i for i, val in enumerate(list(changes_mask)) if val == 1]).to(device=z_t.device)
            num_changes = (old_z_t[sample_in_batch] != z_t[sample_in_batch]).sum().item()
            for j in range(z_t[0, -max_length:].shape[0]):
                puzzle_string += f"{z_t[sample_in_batch, -max_length + j].item()}"
                if j % 9 == 8:
                    puzzle_string += " "
            print(f"num changes: {num_changes}")
            print(f"changes at: {indices_of_change}({indices_of_change - max_length}), old tokens: {torch.gather(old_z_t[sample_in_batch], 0, indices_of_change)}, new tokens: {torch.gather(z_t[sample_in_batch], 0, indices_of_change)}")
            print(f"Step {i}, Puzzle: {puzzle_string}")
            print((f"Fully unmasked: {(z_t[sample_in_batch, -max_length:] != mask_token_id).all()}"))
            if keep_history or debug_flag:
                history.append(z_t.clone())
            if (z_t[:, -max_length:] != mask_token_id).all():
                print(f"All tokens unmasked at step {i}, stopping early.")
                break
        # extra_step_counter = 0
        # mask_tokens_remaining = (z_t[:, -max_length:] == mask_token_id).sum()
        # while (z_t[:, -max_length:] == mask_token_id).any() and extra_step_counter < 10:
        #     extra_step_counter += 1
        #     z_t = self.sampling_step(z_t, ts[0], ts[0], diffusion_mask=diffusion_mask, puzzle_conditioning=puzzle_conditioning, sequence_length_without_conditioning=max_length)
        #     if keep_history or debug_flag:
        #         history.append(z_t.clone())
        # if extra_step_counter > 0:
        #     print(f"{extra_step_counter} extra steps taken to unmask remaining mask tokens, before: {mask_tokens_remaining}, after: {(z_t[:, -max_length:] == mask_token_id).sum()}, of total: {initial_num_mask_tokens}")
        #     # if debug_flag:
        #         # history_tensor = torch.stack(history, dim=0).permute(1, 0, 2)  # (bs, num_denoising_steps + 1, max_length)
        #         # print(f"history: {history_tensor[0, -extra_step_counter:, -max_length:]}")
        if keep_history:
            return z_t, torch.stack(history, dim=0).permute(1, 0, 2) # (bs, num_denoising_steps + 1, max_length)
        else:
            return z_t, None
    
    def _do_generate_from_expected_t(self, initial_z_t, diffusion_mask, num_denoising_steps, add_random_tokens, max_length, show_progress, device, keep_history=False):
        # TODO: when model was trained without diffusion_mask, how do you insert the knowledge of the given puzzle?
        # Idea: pass a modified diffusion_mask to not change anything until the denoising step is reached for which the puzzle is an expected state.
        # With uniform noise, the expected state is never k correct tokens and the rest masked, so potentially randomly unmask some non-given tokens.
        
        # count the number of given tokens
        num_given_tokens = torch.sum(diffusion_mask == 0, dim=-1) - 2 # bos, eos
        seq_len_wo_bos_eos = diffusion_mask.shape[-1] - 2
        
        # compute the denoising step for which the updated initial_z_t is most likely
        # marginal forward distribution: qt(zt|x) = 1 / Ct  ((1 − t)x + tm + ct1).
        # -> num_given_tokens / seq_len = (1 - t) / (1 - t + t + ct * N)? Since 1-t, t and ct * N are the relative weights for denoised, masked and random tokens
        fraction_denoised = num_given_tokens / seq_len_wo_bos_eos
        most_likely_t = self.sampling_step.noise_schedule.get_t(fraction_denoised)
        
        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps
        
        # find the step where the t in the ts array is closest to most_likely_t
        # TODO: put things on the right device at the right time
        ts_expanded= ts.cpu().unsqueeze(0).expand((num_given_tokens.shape[0], -1)) # shape of ts_expanded: (num_samples, num_denoising_steps + 1)
        most_likely_step = torch.argmin(torch.abs(ts_expanded - most_likely_t.unsqueeze(-1)), dim=-1, keepdim=True)
        most_likely_t_discretized = ts.cpu()[most_likely_step].squeeze(-1)
        
        ts = ts.unsqueeze(-1)
        
        if add_random_tokens:
            # randomly unmask some non-given tokens from initial_z_t based on the number of given tokens
            # -> ctN = (1 - t) / (num_given_tokens / seq_len) - 1
            # -> fraction_random = ctN / (1 + ctN)
            ctN = (1 - most_likely_t_discretized) / fraction_denoised - 1
            fraction_random = ctN / (1 + ctN)
            num_random_tokens = (fraction_random * seq_len_wo_bos_eos).to(int)
            # print(f"fraction_denoised: {fraction_denoised},\nmost_likely_step: {most_likely_step},\nnum_random_tokens: {num_random_tokens}")

            initial_z_t_noisy = initial_z_t.clone()
            for i in range(initial_z_t.shape[0]):
                eligible_indices = torch.nonzero(diffusion_mask[i, 1:-1] == 0).squeeze(-1)
                num_to_replace = min(num_random_tokens[i].item(), eligible_indices.numel())
                if num_to_replace > 0:
                    chosen_indices = eligible_indices[torch.randperm(eligible_indices.numel())[:num_to_replace]]
                    initial_z_t_noisy[i, chosen_indices] = torch.randint(0, 9, (num_to_replace,))
            
            initial_z_t_noisy = initial_z_t_noisy.to(device, non_blocking=True)
        
        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        most_likely_step = most_likely_step.to(device, non_blocking=True)
        
        z_t = initial_z_t_noisy.clone() if add_random_tokens else initial_z_t.clone()
        
        history = [z_t.clone()] if keep_history else None
        
        # print("entering sampling loop in _do_generate_from_given")
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            # print(f"sampling step {i}")
            # set the diffusion mask to 0 where the denoising step is not yet reached
            most_likely_step_mask = (i < most_likely_step)
            current_diffusion_mask = diffusion_mask * most_likely_step_mask
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=current_diffusion_mask)
            # print(f"sampling step {i} done")
            if keep_history:
                history.append(z_t.clone())
        if keep_history:
            return z_t, torch.stack(history, dim=0).permute(1, 0, 2) # (bs, num_denoising_steps + 1, max_length)
        else:
            return z_t, None


class MDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, config, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = tokenizer.mask_token_id
            self.min_p = min_p
            # self.position_metric = get_position_metric(config, tokenizer)
            # self.position_sampling_strategy = get_position_sampling_strategy(config)
            # self.token_sampling_strategy = get_token_sampling_strategy(config, tokenizer)
            self.sampling_strategy = get_sampling_strategy(config, tokenizer)
            self.config = config

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, diffusion_mask, i=None, eps=1e-4):
            logits = self.model(z_t, t)
            logits[..., self.mask_id] = -1e6

            update_positions = (z_t == self.mask_id)
            if i == 0:
                z_tm1 = logits.argmax(-1)
            else:
                probs = logits.softmax(-1)

                update_positions_from_strategy, z_tm1 = self.sampling_strategy(probs, z_t, t, tm1, diffusion_mask, eps)
                update_positions = update_positions * update_positions_from_strategy

            update_positions = update_positions * diffusion_mask
            return torch.where(update_positions.bool(), z_tm1, z_t)

    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(config, model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, eps=self.t_eps).clone()

        return z_t
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, puzzle_conditioning, num_denoising_steps, max_length, show_progress, device, keep_history=False):
        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)

        z_t = initial_z_t.clone()

        history = [initial_z_t.clone()] if keep_history else None

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=diffusion_mask, i=i, eps=self.t_eps).clone()
            if keep_history:
                history.append(z_t.clone())
        
        if keep_history:
            return z_t, torch.stack(history, dim=0).permute(1, 0, 2)
        else:
            return z_t, None


class AutoregressiveSampler(Sampler):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True):
        super().__init__(model, tokenizer, noise_schedule)
        if compile_step:
            self.model = torch.compile(model)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full((num_samples, max_length), eos_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long, device=device)
        input_ids[:, 0] = bos_token_id
        attention_mask[:, 0] = 1

        done = torch.zeros(num_samples, device=device)
        for i in tqdm.trange(1, max_length, desc="Generating samples", disable=not show_progress):
            logits = self.model(input_ids, use_cache=False).logits[:, i-1]
            probs = logits.softmax(-1)
            next_x = (1 - done) * sample_categorical(probs) + done * self.tokenizer.pad_token_id
            input_ids[:, i] = next_x.to(input_ids.dtype)
            done += (1 - done) * (next_x == eos_token_id).to(done.dtype)
            if (done == 1).all():
                break

        return input_ids


def get_sampler(ckpt_config, model, tokenizer, noise_schedule: NoiseSchedule, sampling_config=None, compile_step=True, min_p=0.0):
    if sampling_config is None:
        sampling_config = ckpt_config
    if ckpt_config.model.type == "diffusion":
        if ckpt_config.model.diffusion_process == "gidd":
            return GiddSampler(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif ckpt_config.model.diffusion_process == "mdlm":
            return MDLMSampler(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {ckpt_config.model.diffusion_process}")
    elif ckpt_config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {ckpt_config.model.type}")
