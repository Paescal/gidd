from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from gidd.diffusion_process import NoiseSchedule
from gidd.utils import deduplicate, sample_categorical
from gidd.sampling_strategies import get_sampling_strategy, get_sampling_strategy_class
from gidd.self_correction_strategies import self_correction_original, self_correction_original_oscillation_prevention, self_correction_keep_where_confident
from gidd.eval.generation_info import GenerationInfoHandler

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
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, solution, num_denoising_steps, num_self_correction_steps, max_length, show_progress, device, generation_info_handler):
        raise NotImplementedError
    
    @abstractmethod
    def _do_beam_search(self, beam_search_config, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device, generation_info_handler):
        raise NotImplementedError
    
    # Generate starting from a given z_t, the diffusion_mask is a tensor with the same shape as z_t of 0s and 1s, where 1 indicates that the token is NOT given and needs to be denoised
    @torch.no_grad()
    def generate_from_given(self, z_t, diffusion_mask, solution=None, puzzle_conditioning=None, num_denoising_steps=128, num_self_correction_steps=0, max_length=None, decode=True, show_progress=True, generation_info_handler=None, beam_search_config=None):
        max_length = max_length or self.model.config.model.max_seq_len
        # print("getting device")
        device = next(self.model.parameters()).device

        if beam_search_config is not None and beam_search_config.do_beam_search == 'true':
            if z_t.shape[0] != 1:
                raise ValueError("Beam search only supported for batch size 1")
            z_t = self._do_beam_search(beam_search_config, z_t, diffusion_mask=diffusion_mask, solution=solution, puzzle_conditioning=puzzle_conditioning, num_denoising_steps=num_denoising_steps, num_self_correction_steps=num_self_correction_steps, max_length=max_length, device=device, generation_info_handler=generation_info_handler)
        else:
            if puzzle_conditioning is None:
                z_t = self._do_generate_from_given(z_t, diffusion_mask=diffusion_mask, solution=solution, puzzle_conditioning=z_t.clone(), num_denoising_steps=num_denoising_steps, num_self_correction_steps=num_self_correction_steps, max_length=max_length, show_progress=show_progress, device=device, generation_info_handler=generation_info_handler)
            else:
                z_t = self._do_generate_from_given(z_t, diffusion_mask=diffusion_mask, solution=solution, puzzle_conditioning=puzzle_conditioning, num_denoising_steps=num_denoising_steps, num_self_correction_steps=num_self_correction_steps, max_length=max_length, show_progress=show_progress, device=device, generation_info_handler=generation_info_handler)
        # print("done calling _do_generate_from_given")

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            return z_t
    
    @abstractmethod
    def _do_generate_from_expected_t(self, initial_z_t, diffusion_mask, num_denoising_steps, add_random_tokens, max_length, show_progress, device):
        raise NotImplementedError
    
    @torch.no_grad()
    def generate_from_expected_t(self, z_t, diffusion_mask, num_denoising_steps=128, add_random_tokens=True, max_length=None, decode=True, show_progress=True):
        max_length = max_length or self.model.config.max_seq_len
        # print("getting device")
        device = next(self.model.parameters()).device

        # print("calling _do_generate_from_expected_t")
        z_t = self._do_generate_from_expected_t(z_t, diffusion_mask=diffusion_mask, num_denoising_steps=num_denoising_steps, add_random_tokens=add_random_tokens, max_length=max_length, show_progress=show_progress, device=device)
        # print("done calling _do_generate_from_expected_t")

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
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
            self.max_score = None

        def forward(self, z_t, t, s, i=None, num_denoising_steps=None, diffusion_mask=None, puzzle_conditioning=None, sequence_length_without_conditioning=0):
            is_fully_denoised = (z_t[:, -sequence_length_without_conditioning:] != self.tokenizer.mask_token_id).all(dim=1)
            # if is_fully_denoised.sum() != 0:
            #     print(f"is_fully_denoised: {is_fully_denoised.sum()}")
            # print("inside gidd denoising step")
            logits = self.model(z_t, t, puzzle_conditioning=puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)
            # if i == 77: # debugging for checkpoints/gidd_0_2/100_epochs,gidd_keep_where_confident,"score_position_for_change=change_max select_position=top_k_gumbel change_token=change_max k=1 gumbel_noise_coefficient=0 self_correction=none dataset=hard num_samples=64 num_denoising_steps=81 batch_size=64 min_p=0 compile_torch=0 seed=1"
            #     print(f"probs: {probs[4, 123, :9]}")
            #     print(f"logits: {logits[4, 123, :9]}")
            #     print(f"t: {t}")
            #     print(f"z_t: {z_t[4]}")
            if self.config.sampling.strategy == "gidd_flattened":
                if self.max_score is None:
                    self.max_score = torch.zeros_like(z_t, dtype=probs.dtype)
                update_positions, next_z_t, max_score = self.sampling_strategy(probs, z_t, t, s, i, num_denoising_steps, diffusion_mask, self.max_score)
                self.max_score = max_score
            else:
                update_positions, next_z_t = self.sampling_strategy(probs, z_t, t, s, i, diffusion_mask)
            # update_positions, next_z_t = self.sampling_strategy(probs, z_t, t, s, diffusion_mask)
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
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, show_progress, device, generation_info_handler=None):
        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps
        
        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        
        z_t = initial_z_t.clone()
        
        # print("entering sampling loop in _do_generate_from_given")
        mask_token_id = self.sampling_step.tokenizer.mask_token_id
        # initial_num_mask_tokens = (z_t[:, -max_length:] == mask_token_id).sum()
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            # print(f"sampling step {i}")
            # old_z_t = z_t.clone()
            # if i > num_denoising_steps - 5: # debugging for checkpoints/gidd_0_2/100_epochs,gidd_keep_where_confident,"score_position_for_change=change_max select_position=top_k_gumbel change_token=change_max k=1 gumbel_noise_coefficient=0 self_correction=none dataset=hard num_samples=64 num_denoising_steps=81 batch_size=64 min_p=0 compile_torch=0 seed=1"
            #     print(i, z_t[4, 123].item())
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, num_denoising_steps=num_denoising_steps, diffusion_mask=diffusion_mask, puzzle_conditioning=puzzle_conditioning, sequence_length_without_conditioning=max_length)
            # if i > num_denoising_steps - 5: # debugging for checkpoints/gidd_0_2/100_epochs,gidd_keep_where_confident,"score_position_for_change=change_max select_position=top_k_gumbel change_token=change_max k=1 gumbel_noise_coefficient=0 self_correction=none dataset=hard num_samples=64 num_denoising_steps=81 batch_size=64 min_p=0 compile_torch=0 seed=1"
            #     print(i, z_t[4, 123].item())
            # print(f"sampling step {i} done")
            
            # puzzle_string = ''
            # sample_in_batch = 31
            # changes_mask = (old_z_t[sample_in_batch] != z_t[sample_in_batch]).to(dtype=int)
            # indices_of_change = torch.tensor([i for i, val in enumerate(list(changes_mask)) if val == 1]).to(device=z_t.device)
            # num_changes = (old_z_t[sample_in_batch] != z_t[sample_in_batch]).sum().item()
            # for j in range(z_t[0, -max_length:].shape[0]):
            #     puzzle_string += f"{z_t[sample_in_batch, -max_length + j].item()}"
            #     if j % 9 == 8:
            #         puzzle_string += " "
            # print(f"num changes: {num_changes}")
            # print(f"changes at: {indices_of_change}({indices_of_change - max_length}), old tokens: {torch.gather(old_z_t[sample_in_batch], 0, indices_of_change)}, new tokens: {torch.gather(z_t[sample_in_batch], 0, indices_of_change)}")
            # print(f"Step {i}, Puzzle: {puzzle_string}")
            # print((f"Fully unmasked: {(z_t[sample_in_batch, -max_length:] != mask_token_id).all()}"))
            # if (z_t[:, -max_length:] != mask_token_id).all():
            #     print(f"All tokens unmasked at step {i}, stopping early.")
            #     break
        
        self_correction = self.sampling_step.config.sampling.self_correction
        if self_correction == "none":
            history_self_correction = []
        elif self_correction == "original":
            temp = 1
            tokens_per_step = 1
            z_t = self_correction_original(self.model, self.tokenizer, diffusion_mask, z_t, ts[0].item(), temp, tokens_per_step, max_num_denoising_steps=num_self_correction_steps)
        elif self_correction == "oscillation_prevention_fast":
            temp = 1
            tokens_per_step = 1
            backoff_factor=0.5
            recovery_factor_fast=0.2
            recovery_type='fast'
            z_t = self_correction_original_oscillation_prevention(self.model, self.tokenizer, diffusion_mask, z_t, ts[0].item(), temp, tokens_per_step, max_num_denoising_steps=num_self_correction_steps, backoff_factor=backoff_factor, recovery_factor_fast=recovery_factor_fast, recovery_type=recovery_type)
        elif self_correction == "oscillation_prevention_slow":
            temp = 1
            tokens_per_step = 1
            backoff_factor=0.5
            recovery_factor_slow=0.4142
            recovery_type='slow'
            z_t = self_correction_original_oscillation_prevention(self.model, self.tokenizer, diffusion_mask, z_t, ts[0].item(), temp, tokens_per_step, max_num_denoising_steps=num_self_correction_steps, backoff_factor=backoff_factor, recovery_factor_slow=recovery_factor_slow, recovery_type=recovery_type)
        elif self_correction == "keep_where_confident":
            temp = 1
            tokens_per_step = 1
            z_t = self_correction_keep_where_confident(self.model, self.tokenizer, diffusion_mask, z_t, ts[0].item(), temp, tokens_per_step, max_num_denoising_steps=num_self_correction_steps)
        # elif self_correction == "max":
        #     temp = 1
        #     tokens_per_step = 1
        #     z_t = self_correction_max(self.model, self.tokenizer, diffusion_mask, z_t, ts[0].item(), temp, tokens_per_step)
        
        # print(f"fully unmasked samples: {(z_t[:, -max_length:] != mask_token_id).all(dim=1)}")
        
        # extra_step_counter = 0
        # mask_tokens_remaining = (z_t[:, -max_length:] == mask_token_id).sum()
        # while (z_t[:, -max_length:] == mask_token_id).any() and extra_step_counter < 10:
        #     extra_step_counter += 1
        #     z_t = self.sampling_step(z_t, ts[0], ts[0], diffusion_mask=diffusion_mask, puzzle_conditioning=puzzle_conditioning, sequence_length_without_conditioning=max_length)
        # if extra_step_counter > 0:
        #     print(f"{extra_step_counter} extra steps taken to unmask remaining mask tokens, before: {mask_tokens_remaining}, after: {(z_t[:, -max_length:] == mask_token_id).sum()}, of total: {initial_num_mask_tokens}")
        #     # if debug_flag:
        #         # history_tensor = torch.stack(history, dim=0).permute(1, 0, 2)  # (bs, num_denoising_steps + 1, max_length)
        #         # print(f"history: {history_tensor[0, -extra_step_counter:, -max_length:]}")
        return z_t
    
    def _do_generate_from_expected_t(self, initial_z_t, diffusion_mask, num_denoising_steps, add_random_tokens, max_length, show_progress, device):
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
        
        # print("entering sampling loop in _do_generate_from_given")
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            # print(f"sampling step {i}")
            # set the diffusion mask to 0 where the denoising step is not yet reached
            most_likely_step_mask = (i < most_likely_step)
            current_diffusion_mask = diffusion_mask * most_likely_step_mask
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=current_diffusion_mask)
            # print(f"sampling step {i} done")
        return z_t

class GiddSampler_new(Sampler):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_strategy = get_sampling_strategy_class(config, model, noise_schedule, tokenizer, self.t_eps, min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):

        # ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        # ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        # # zt = sample_categorical(p_zt)
        # z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        # for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
        #     z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        # return z_t
        raise NotImplementedError
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, show_progress, device, generation_info_handler:GenerationInfoHandler=None):
        self.sampling_strategy.initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        if generation_info_handler is not None:
            generation_info_handler.batch_initialize(initial_z_t, diffusion_mask, solution, device)
        # history = [initial_z_t.clone().to(device, non_blocking=True)] if collect_history else None
        for i in tqdm.trange(num_denoising_steps + num_self_correction_steps, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            if self.sampling_strategy.stopping_criterion(i, generation_info_handler):
                # print(f"Stopping criterion met at step {i}")
                break
            self.sampling_strategy.step(i, generation_info_handler)
            # if collect_history:
            #     history.append(self.sampling_strategy.z_t.clone())
        if generation_info_handler is not None:
            generation_info_handler.batch_finalize()
        return self.sampling_strategy.z_t
        # if collect_history:
        #     return self.sampling_strategy.z_t, torch.stack(history, dim=0).permute(1, 0, 2) # (num_denoising_steps + self_correction_steps + 1, bs, max_length) -> (bs, num_denoising_steps + self_correction_steps + 1, max_length)
        # else:
        #     return self.sampling_strategy.z_t, []
    
    def _do_beam_search(self, beam_search_config, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device, generation_info_handler:GenerationInfoHandler=None):
        def check_beam_correct(beam):
            return (torch.logical_or(beam['z_t'] == solution, beam['z_t'] == self.tokenizer.mask_token_id) * diffusion_mask.bool()).sum().item() == torch.sum(diffusion_mask).item()
        # while beams exist with denoising_progress < max_denoising progress (=not done)
        #   while beams exist with steps_before_pruning > 0
        #       progress them by one step (branch by branching_factor)
        #   deduplicate beams
        #   once all beams have steps_before_pruning == 0:
        #   compute min_progress = min(denoising_progress)
        #   select the beams with denoising_progress == min_progress
        #   prune selected beams to top pruning_num_beams beams
        #   reset steps_before_pruning for pruned beams

        # Current assumptions: batch size == 1

        if generation_info_handler is not None:
            generation_info_handler.batch_initialize(initial_z_t, diffusion_mask, solution, device)

        max_num_steps = num_denoising_steps
        steps_before_pruning = beam_search_config.steps_before_pruning
        pruning_num_beams = beam_search_config.pruning_num_beams
        branching_factor = beam_search_config.branching_factor
        score_time = beam_search_config.score_time
        score_method = beam_search_config.score_method
        initiate_beam_search_after_progress = beam_search_config.initiate_beam_search_after_progress
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        solution = solution.to(device, non_blocking=True)
        final_denoising_progress = torch.sum(diffusion_mask).item()
        progress_threshold_to_branch = int(final_denoising_progress * initiate_beam_search_after_progress)
        progress_threshold_to_branch_reached = False

        beams = []
        complete_beams = []
        self.sampling_strategy.initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        initial_beam = self.sampling_strategy.get_state()
        initial_beam['steps_before_pruning'] = steps_before_pruning
        initial_beam['denoising_progress'] = 0 # number of tokens unmasked?
        initial_beam['step'] = 0
        initial_beam['parent_score'] = 0
        beams.append(initial_beam)
        
        beam_search_complete = False
        pruned_last_iteration = False
        while not beam_search_complete:
            next_beams = []
            beam_search_complete = True
            ready_for_pruning = True
            for beam in beams:
                if beam['denoising_progress'] == final_denoising_progress and not pruned_last_iteration:
                    if self.sampling_strategy.beam_score_final_accepted(beam, generation_info_handler):
                        if generation_info_handler is not None:
                            generation_info_handler.batch_finalize()
                        return beam['z_t']
                if beam['step'] >= max_num_steps:
                    complete_beams.append(beam)
                    continue
                beam_search_complete = False
                if beam['steps_before_pruning'] > 0:
                    ready_for_pruning = False
                    if not progress_threshold_to_branch_reached and beam['denoising_progress'] >= progress_threshold_to_branch:
                        progress_threshold_to_branch_reached = True
                        if generation_info_handler is not None:
                            generation_info_handler.beam_search_initial_beam_correct_step(check_beam_correct(beam))

                    if beam['denoising_progress'] >= progress_threshold_to_branch and beam['steps_before_pruning'] == steps_before_pruning:
                        current_branching_factor = branching_factor
                    else:
                        current_branching_factor = 1
                    new_beams = self.sampling_strategy.branch_step(beam, current_branching_factor, generation_info_handler)
                    num_correct = 0
                    for new_beam in new_beams:
                        new_beam['steps_before_pruning'] = beam['steps_before_pruning'] - 1
                        new_beam['denoising_progress'] = torch.sum((new_beam['z_t'] != self.tokenizer.mask_token_id) * diffusion_mask).item()
                        new_beam['step'] = beam['step'] + 1
                        # for scoring relative to parent:
                        new_beam['parent_score'] = beam['parent_score']
                        if check_beam_correct(new_beam):
                            num_correct += 1
                    # print(f"Step {beam['step']}: branched into {len(new_beams)} beams, {num_correct} correct")
                    old_beam_is_correct = check_beam_correct(beam)
                    # if old_beam_is_correct and num_correct == 0:
                    #     print(f'Step {beam["step"]}: branched out of a correct beam but none of the new beams are correct!')
                    # elif not old_beam_is_correct and num_correct > 0:
                    #     print(f'Step {beam["step"]}: branched into {num_correct} correct beams from an incorrect beam!')
                    next_beams = next_beams + new_beams
                else:
                    next_beams.append(beam)
            # if not ready_for_pruning:
            #     print(f"Before deduplication: {len(next_beams)} beams")
            next_beams = deduplicate(next_beams)
            if not ready_for_pruning:
                pruned_last_iteration = False
                num_correct = [check_beam_correct(beam) for beam in next_beams].count(True)
                # print(f"After deduplication: {len(next_beams)} beams, {num_correct} correct")
            if ready_for_pruning and not beam_search_complete:
                pruned_last_iteration = True
                # perform pruning
                # min_progress = min([beam['denoising_progress'] for beam in next_beams])
                # beams_to_prune = [beam for beam in next_beams if beam['denoising_progress'] == min_progress]
                min_progress = min([beam['step'] for beam in next_beams])
                beams_to_prune = [beam for beam in next_beams if beam['step'] == min_progress]
                num_correct_before_pruning = [check_beam_correct(beam) for beam in beams_to_prune].count(True)
                num_incorrect_before_pruning = len(beams_to_prune) - num_correct_before_pruning
                if len(beams_to_prune) > pruning_num_beams:
                    # compute scores for beams to prune
                    beam_scores = []
                    for beam in beams_to_prune:
                        score = self.sampling_strategy.beam_score(beam, score_time, score_method, generation_info_handler)
                        beam_scores.append(score)
                    # for scoring relative to parent:
                    relative_beam_scores = [beam_scores[i] - beams_to_prune[i]['parent_score'] for i in range(len(beam_scores))]
                    for i, beam in enumerate(beams_to_prune):
                        beam['parent_score'] = beam_scores[i]
                    beam_scores = relative_beam_scores
                    # select top pruning_num_beams beams
                    topk_indices = torch.topk(torch.tensor(beam_scores), k=min(pruning_num_beams, len(beams_to_prune))).indices.tolist()
                    pruned_beams = [beams_to_prune[i] for i in topk_indices]
                else:
                    pruned_beams = beams_to_prune
                # reset steps_before_pruning
                for beam in pruned_beams:
                    beam['steps_before_pruning'] = steps_before_pruning
                num_correct_after_pruning = [check_beam_correct(beam) for beam in pruned_beams].count(True)
                num_incorrect_after_pruning = len(pruned_beams) - num_correct_after_pruning
                if generation_info_handler is not None:
                    if num_correct_before_pruning > 0 and num_correct_after_pruning == 0:
                        is_beam_correct = [check_beam_correct(beam) for beam in beams_to_prune]
                        generation_info_handler.prune_correct_step({
                            'z_ts': [beam['z_t'].squeeze(0) for beam in beams_to_prune],
                            'beam_scores': beam_scores,
                            'is_correct': is_beam_correct,
                            'solution': solution.squeeze(0),
                            'steps': [beam['step'] for beam in beams_to_prune],
                        })
                        # print(f'Step {beam["step"]}: pruned out all correct beams! beam scores: {beam_scores}, correct beam? {is_beam_correct}')
                    generation_info_handler.beam_search_branch_correctness_step(min_progress, num_correct_before_pruning, num_incorrect_before_pruning, num_correct_after_pruning, num_incorrect_after_pruning)
                # add pruned beams back to next_beams
                # next_beams = [beam for beam in next_beams if beam['denoising_progress'] != min_progress] + pruned_beams
                next_beams = [beam for beam in next_beams if beam['step'] != min_progress] + pruned_beams
                # print(f"After pruning: {len(next_beams)} beams, {[(torch.logical_or(beam['z_t'] == solution, beam['z_t'] == self.tokenizer.mask_token_id) * diffusion_mask.bool()).sum().item() == torch.sum(diffusion_mask).item() for beam in next_beams].count(True)} correct")
            beams = next_beams
        # select best complete beam
        complete_beams = deduplicate(complete_beams)
        best_beam = None
        best_score = -float('inf')
        for beam in complete_beams:
            score = self.sampling_strategy.beam_score_final(beam, generation_info_handler) # may want to use a special score function for completed beams (e.g. min confidence over all tokens)
            if score > best_score:
                best_score = score
                best_beam = beam
        # if (best_beam['z_t'] != solution).any():
        #     print(f'failed to find correct solution in beam search, best score: {best_score}')
        if generation_info_handler is not None:
            generation_info_handler.batch_finalize()
        return best_beam['z_t']

class MDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, config, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = tokenizer.mask_token_id
            self.min_p = min_p
            self.sampling_strategy = get_sampling_strategy(config, tokenizer)
            self.config = config

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, diffusion_mask, i=None, eps=1e-4, generation_info_handler=None):
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
            if generation_info_handler is not None:
                generation_info_handler.batch_step_logits(logits[..., :self.mask_id])
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
    
    def _do_generate_from_given(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, show_progress, device, generation_info_handler=None):
        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        initial_z_t = initial_z_t.to(device, non_blocking=True)
        diffusion_mask = diffusion_mask.to(device, non_blocking=True)

        z_t = initial_z_t.clone()

        if generation_info_handler is not None:
            generation_info_handler.batch_initialize(initial_z_t, diffusion_mask, solution, device)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            if generation_info_handler is not None:
                generation_info_handler.batch_step_history(z_t)
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], diffusion_mask=diffusion_mask, i=i, eps=self.t_eps, generation_info_handler=generation_info_handler).clone()
        if generation_info_handler is not None:
            generation_info_handler.batch_step_history(z_t)
            generation_info_handler.batch_finalize()
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
            # return GiddSampler(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
            return GiddSampler_new(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif ckpt_config.model.diffusion_process == "mdlm":
            return GiddSampler_new(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
            # return MDLMSampler(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, compile_step=compile_step, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {ckpt_config.model.diffusion_process}")
    elif ckpt_config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {ckpt_config.model.type}")
