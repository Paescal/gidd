from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from gidd.diffusion_process import NoiseSchedule
from gidd.utils import get_position_metric, get_position_sampling_strategy, get_token_sampling_strategy, sample_categorical


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
    def generate_from_given(self, z_t, diffusion_mask, num_denoising_steps=128, max_length=None, decode=True, show_progress=True, keep_history=False):
        max_length = max_length or self.model.config.model.max_seq_len
        if self.model.config.model.use_puzzle_conditioning:
            max_length = 2 * max_length
        # print("getting device")
        device = next(self.model.parameters()).device

        # print("calling _do_generate_from_given")
        z_t, history = self._do_generate_from_given(z_t, diffusion_mask=diffusion_mask, num_denoising_steps=num_denoising_steps, max_length=max_length, show_progress=show_progress, device=device, keep_history=keep_history)
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
            self.position_metric = get_position_metric(config, tokenizer)
            self.position_sampling_strategy = get_position_sampling_strategy(config)
            self.token_sampling_strategy = get_token_sampling_strategy(config, tokenizer)
            self.config = config

        def forward(self, z_t, t, s, diffusion_mask=None):
            # print("inside gidd denoising step")
            logits = self.model(z_t, t)
            # print("got logits from model")
            # print(f"logits: {logits[..., self.tokenizer.mask_token_id - 1]}")
            logits[..., self.tokenizer.mask_token_id:] = -1e6

            # if i > 0:
            # print("getting probs at t and s")
            q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
            q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
            q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            # print("getting alpha and beta pi at t and s")
            alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            # vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
            vocab_size_architecturally = len(self.tokenizer)
            vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
            beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

            q_st = q_ts * q_s / q_zt
            if self.min_p > 0.0:
                is_small = (q_st < self.min_p).float()
                q_st = (1 - is_small) * q_st
                q_st = q_st / q_st.sum(-1, keepdim=True)
            # print(f"z_t: {z_t[..., 1]}")
            # print(f"q_s: {q_s[..., 1, :10]}")
            # print(f"q_st: {q_st[..., 1, :10]}")
            # print("getting metric")
            # print(f"probs of mask token: {q_st[..., self.tokenizer.mask_token_id]}")
            metric = self.position_metric(z_t, q_st)
            # print(f"metric: {metric[..., 1]}")
            metric = metric * diffusion_mask
            # print("getting update positions")
            if self.config.sampling.position_sampling_strategy == "independent":
                update_positions = self.position_sampling_strategy(metric, (alpha_s - alpha_t) / (1 - alpha_t))
            else:
                update_positions = self.position_sampling_strategy(metric)
            update_positions = update_positions * diffusion_mask
            # print("getting next z_t")
            next_z_t = self.token_sampling_strategy(q_st)
            # print(f"z_t: {z_t}")
            # print(f"update_positions: {update_positions}")
            # print(f"next_z_t: {next_z_t}")
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
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, num_denoising_steps, max_length, show_progress, device, keep_history=False):
        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps
        
        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        
        z_t = initial_z_t.clone()
        
        history = [initial_z_t.clone()] if keep_history else None
        
        # print("entering sampling loop in _do_generate_from_given")
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            # print(f"sampling step {i}")
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=diffusion_mask)
            # print(f"sampling step {i} done")
            if keep_history:
                history.append(z_t.clone())
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
        def __init__(self, model, noise_schedule, mask_id, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = mask_id
            self.min_p = min_p

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, i=None, eps=1e-4):
            logits = self.model(z_t, t)
            logits[..., self.mask_id] = -1e6

            if i == 0:
                z_tm1 = logits.argmax(-1)
            else:
                _, sigma_t = self.get_sigmas(t, eps=eps)
                _, sigma_tm1 = self.get_sigmas(tm1, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]
                probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                probs /= move_chance_t
                if self.min_p > 0.0:
                    is_small = (probs < self.min_p).float()
                    probs = (1 - is_small) * probs
                    probs = probs / probs.sum(-1, keepdim=True)
                z_tm1 = sample_categorical(probs)
                # z_tm1 = torch.distributions.Categorical(probs=probs).sample()
                # z_tm1 = _sample_categorical(probs)

            copy_flag = (z_t != self.mask_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1
            return z_t

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer.mask_token_id, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, eps=self.t_eps).clone()

        return z_t


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
            return MDLMSampler(model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {ckpt_config.model.diffusion_process}")
    elif ckpt_config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {ckpt_config.model.type}")
