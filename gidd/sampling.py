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

class GiddSampler_new(Sampler):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_strategy = get_sampling_strategy_class(config, model, noise_schedule, tokenizer, self.t_eps, min_p)

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
        initial_beam['beam_id'] = 0
        score_relative_to_parent = False
        if score_relative_to_parent:
            initial_beam['parent_score'] = 0
        beams.append(initial_beam)
        
        beam_search_complete = False
        pruned_last_iteration = False
        beam_id_counter = 1
        beam_started_with_correct_branches = {}
        beam_started_with_correct_branches_intermediate = {}
        while not beam_search_complete:
            next_beams = []
            beam_search_complete = True
            ready_for_pruning = True
            beam_has_correct_branches = {}
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
                    if current_branching_factor > 1:
                        beam_id_counter += 1
                        current_beam_id = beam_id_counter
                    else:
                        current_beam_id = beam['beam_id']
                    new_beams = self.sampling_strategy.branch_step(beam, current_branching_factor, generation_info_handler)
                    num_correct = 0
                    for new_beam in new_beams:
                        new_beam['steps_before_pruning'] = beam['steps_before_pruning'] - 1
                        new_beam['denoising_progress'] = torch.sum((new_beam['z_t'] != self.tokenizer.mask_token_id) * diffusion_mask).item()
                        new_beam['step'] = beam['step'] + 1
                        new_beam['beam_id'] = current_beam_id
                        beam_has_correct_branches[current_beam_id] = beam_has_correct_branches.get(current_beam_id, False) or check_beam_correct(new_beam)
                        if score_relative_to_parent:
                            new_beam['parent_score'] = beam['parent_score']
                        if check_beam_correct(new_beam):
                            num_correct += 1
                    # print(f"Step {beam['step']}: branched into {len(new_beams)} beams, {num_correct} correct")
                    if current_branching_factor > 1:
                        new_beam_started_with_correct = beam_has_correct_branches[current_beam_id]
                        beam_started_with_correct_branches[current_beam_id] = new_beam_started_with_correct
                        beam_started_with_correct_branches_intermediate[current_beam_id] = new_beam_started_with_correct
                        parent_beam_correct = check_beam_correct(beam)
                        if not new_beam_started_with_correct and parent_beam_correct:
                            # a correct beam branched into incorrect beams only
                            generation_info_handler.beam_search_beam_correctness_step('bad_branch')
                        if new_beam_started_with_correct and not parent_beam_correct:
                            # an incorrect beam branched into some correct beams
                            generation_info_handler.beam_search_beam_correctness_step('good_branch')
                    # if not generation_info_handler is None:
                    #     old_beam_is_correct = check_beam_correct(beam)
                    #     if old_beam_is_correct and num_correct == 0:
                    #         # generation_info_handler.bad_branch_step()
                    #         if current_branching_factor > 1:
                    #             print(f'Step {beam["step"]}: branched out of a correct beam but none of the new beams are correct!')
                    #         # else:
                    #         #     print(f'Step {beam["step"]}: progressed a correct beam but the new beam is not correct!')
                    #     elif not old_beam_is_correct and num_correct > 0:
                    #         if current_branching_factor > 1:
                    #             print(f'Step {beam["step"]}: branched into {num_correct} correct beams from an incorrect beam!')
                    #         # else:
                    #         #     print(f'Step {beam["step"]}: progressed into a correct beam from an incorrect beam!')
                    next_beams = next_beams + new_beams
                else:
                    current_beam_id = beam['beam_id']
                    beam_has_correct_branches[current_beam_id] = beam_has_correct_branches.get(current_beam_id, False) or check_beam_correct(beam)
                    next_beams.append(beam)
            
            if generation_info_handler is not None:
                for current_beam_id, has_correct_branches in beam_has_correct_branches.items():
                    started_with_correct_intermediate = beam_started_with_correct_branches_intermediate[current_beam_id]
                    if started_with_correct_intermediate and not has_correct_branches:
                        # an initially correct beam lost all correct branches during intermediate propagation
                        generation_info_handler.beam_search_beam_correctness_step('bad_intermediate_propagation')
                        beam_started_with_correct_branches_intermediate[current_beam_id] = False
                    if not started_with_correct_intermediate and has_correct_branches:
                        # an initially incorrect beam gained some correct branches during intermediate propagation
                        generation_info_handler.beam_search_beam_correctness_step('good_intermediate_propagation')
                        beam_started_with_correct_branches_intermediate[current_beam_id] = True
            # if not ready_for_pruning:
            #     print(f"Before deduplication: {len(next_beams)} beams")
            next_beams = deduplicate(next_beams)
            if not ready_for_pruning:
                pruned_last_iteration = False
                # num_correct = [check_beam_correct(beam) for beam in next_beams].count(True)
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
                    if score_relative_to_parent:
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
                        # print(f'Step {beam["step"]}: pruned out all correct beams! (there were {num_correct_before_pruning}/{num_incorrect_before_pruning} correct/incorrect beams before pruning down to {pruning_num_beams} beams)')
                    generation_info_handler.beam_search_branch_correctness_step(min_progress, num_correct_before_pruning, num_incorrect_before_pruning, num_correct_after_pruning, num_incorrect_after_pruning)
                # add pruned beams back to next_beams
                # next_beams = [beam for beam in next_beams if beam['denoising_progress'] != min_progress] + pruned_beams
                if generation_info_handler is not None:
                    beam_ids_pruning = set([beam['beam_id'] for beam in beams_to_prune])
                    for current_beam_id in beam_ids_pruning:
                        if beam_started_with_correct_branches[current_beam_id] and not beam_has_correct_branches[current_beam_id]:
                            # a beam with initially correct branches lost them during propagation by the time of pruning
                            generation_info_handler.beam_search_beam_correctness_step('bad_propagation')
                        if not beam_started_with_correct_branches[current_beam_id] and beam_has_correct_branches[current_beam_id]:
                            # a beam with initially incorrect branches gained some correct branches during propagation by the time of pruning
                            generation_info_handler.beam_search_beam_correctness_step('good_propagation')
                    num_correct_beams_missed = min(pruning_num_beams, num_correct_before_pruning) - num_correct_after_pruning
                    if num_correct_beams_missed > 0:
                        # missed num_correct_beams_missed correct beams during pruning
                        generation_info_handler.beam_search_beam_correctness_step('bad_pruning_branches_missed', num_correct_beams_missed)
                        generation_info_handler.beam_search_beam_correctness_step('bad_pruning_instances')
                    for current_beam_id in beam_ids_pruning:
                        beam_started_with_correct_branches.pop(current_beam_id, None)
                        beam_started_with_correct_branches_intermediate.pop(current_beam_id, None)

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


def get_sampler(ckpt_config, model, tokenizer, noise_schedule: NoiseSchedule, sampling_config=None, min_p=0.0):
    if sampling_config is None:
        sampling_config = ckpt_config
    if ckpt_config.model.type == "diffusion":
        if ckpt_config.model.diffusion_process == "gidd":
            return GiddSampler_new(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, min_p=min_p)
        elif ckpt_config.model.diffusion_process == "mdlm":
            return GiddSampler_new(sampling_config, model, tokenizer, noise_schedule, t_eps=ckpt_config.model.t_eps, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {ckpt_config.model.diffusion_process}")
    else:
        raise ValueError(f"Unsupported model type: {ckpt_config.model.type}")
