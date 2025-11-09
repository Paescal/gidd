from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from gidd.utils import (
    score_position_for_change_max,
    score_position_for_change_margin,
    score_position_for_keep_where_confident_max,
    score_position_for_keep_where_confident_margin,
    score_mask_position_max,
    score_mask_position_margin,
    all_positions,
    sample_positions_independently,
    sample_position_top_k_gumbel,
    sample_position_top_variable_k,
    sample_token_change_max,
    sample_token_change_categorical,
    sample_token_MDM_max,
    sample_token_MDM_categorical,
    sample_categorical,
    mean_over_diffusion_positions,
)
from gidd.self_correction_strategies import get_self_correction

def get_score_mask_position(config, tokenizer):
    match config.score_mask_position:
        case "MDM_max":
            return partial(score_mask_position_max, tokenizer=tokenizer)
        case "MDM_margin":
            return partial(score_mask_position_margin, tokenizer=tokenizer)

def get_score_position_for_change(config):
    match config.score_position_for_change:
        case "change_max":
            return partial(score_position_for_change_max)
        case "change_margin":
            return partial(score_position_for_change_margin)

def get_score_position_for_keep_where_confident(config):
    match config.score_position_for_change:
        case "change_max":
            return partial(score_position_for_keep_where_confident_max)
        case "change_margin":
            return partial(score_position_for_keep_where_confident_margin)

def get_select_position(config, arg_name='select_position'):
    match arg_name:
        case 'select_position':
            arg = config.select_position
        case 'select_position_change':
            arg = config.select_position_change
        case 'select_position_unmask':
            arg = config.select_position_unmask
        case _:
            raise ValueError(f"Unknown argument name: {arg_name}")

    match arg:
        case "all":
            return all_positions # == where metric is not zero
        case "independent":
            return sample_positions_independently
        case "top_k_gumbel":
            return partial(sample_position_top_k_gumbel, k=config.k, gumbel_noise_coefficient=config.gumbel_noise_coefficient)

def get_unmask_token(config, tokenizer):
    match config.unmask_token:
        case "MDM_max":
            return partial(sample_token_MDM_max)
        case "MDM_categorical":
            return partial(sample_token_MDM_categorical, tokenizer=tokenizer)

def get_change_token(config, tokenizer):
    match config.change_token:
        case "change_max":
            return partial(sample_token_change_max)
        case "change_categorical":
            return partial(sample_token_change_categorical, tokenizer=tokenizer)

def get_sampling_strategy(config, tokenizer, noise_schedule=None, min_p=None):
    match config.strategy:
        case "mdlm_vanilla":
            return partial(mdlm_vanilla,
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "mdlm_adaptive_score_select_update":
            return partial(mdlm_adaptive_score_select_update,
                           score_mask_position=get_score_mask_position(config, tokenizer),
                           select_position=get_select_position(config),
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "gidd_emulate_mdlm_vanilla": # gidd_vanilla_split
            return partial(gidd_emulate_mdlm_vanilla,
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "gidd_emulate_mdlm_adaptive_score_select_update": # gidd_adaptive_score_select_update
            return partial(gidd_emulate_mdlm_adaptive_score_select_update,
                           tokenizer=tokenizer,
                           score_mask_position=get_score_mask_position(config, tokenizer),
                           select_position=get_select_position(config),
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "gidd_original":
            return partial(gidd_original,
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,)
        case "gidd_independent_positions_decomposed_update_distribution": # case "gidd_vanilla_independent_change":
            return partial(gidd_independent_positions_decomposed_update_distribution,
                           change_token=get_change_token(config, tokenizer),
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,)
        case "gidd_selected_positions_decomposed_update_distribution": # case "gidd_adaptive_change_vs_unmask":
            return partial(gidd_selected_positions_decomposed_update_distribution,
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,
                           score_mask_position=get_score_mask_position(config, tokenizer),
                           select_position_change=get_select_position(config, arg_name='select_position_change'),
                           select_position_unmask=get_select_position(config, arg_name='select_position_unmask'),
                           change_token=get_change_token(config, tokenizer),
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "gidd_change_based_on_model_confidence_to_change":
            return partial(gidd_change_based_on_model_confidence_to_change,
                           tokenizer=tokenizer,
                           score_position_for_change=get_score_position_for_change(config),
                           select_position=get_select_position(config),
                           change_token=get_change_token(config, tokenizer),)
        case "gidd_keep_where_confident":
            return partial(gidd_keep_where_confident,
                           tokenizer=tokenizer,
                           score_position_for_keep_where_confident=get_score_position_for_keep_where_confident(config),
                           select_position=get_select_position(config),
                           change_token=get_change_token(config, tokenizer),)
        case "gidd_change_low_confidence_positions":
            return partial(gidd_change_low_confidence_positions,
                           tokenizer=tokenizer,
                           score_mask_position=get_score_mask_position(config, tokenizer),
                           select_position_unmask=get_select_position(config, arg_name='select_position_unmask'),
                           change_token=get_change_token(config, tokenizer),
                           unmask_token=get_unmask_token(config, tokenizer),)
        case "gidd_change_lowest_confidence_position":
            return partial(gidd_change_lowest_confidence_position,
                           tokenizer=tokenizer,
                           score_mask_position=get_score_mask_position(config, tokenizer),
                           select_position_unmask=get_select_position(config, arg_name='select_position_unmask'),
                           unmask_token=get_unmask_token(config, tokenizer),
                           k=config.k,)
        case "gidd_flattened":
            return partial(gidd_flattened,
                           tokenizer=tokenizer,)

#################### MDLM sampling strategies ####################
# code for when using GiddSampler_new for mdlm strategies
@torch.no_grad()
def mdlm_vanilla(probs, z_t, t, tm1, i, diffusion_mask, eps=1e-4, unmask_token=None):
    def get_sigmas(t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * torch.clamp(t, eps, 1))
        dsigma = (1 - eps) / (1 - (1 - eps) * torch.clamp(t, eps, 1))
        sigma = -torch.log1p(-(1 - eps) * torch.clamp(t, eps, 1))
        return dsigma, sigma
    # In train for the worst paper, the vanilla inference uses alpha_s and alpha_t. move_chance_tm1 is equal to 1 - alpha_s, move_chance_t is equal to 1 - alpha_t
    # The weight (move_chance_t - move_chance_tm1) / move_chance_t is equal to the probability (alpha_s - alpha_t) / (1 - alpha_t) to select a token for unmasking
    _, sigma_t = get_sigmas(t, eps=eps)
    _, sigma_tm1 = get_sigmas(tm1, eps=eps)
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
    prob_unmask_this_step = (move_chance_t - move_chance_tm1) / move_chance_t
    update_positions = sample_positions_independently(z_t, prob_unmask_this_step.unsqueeze(-1))
    z_tm1 = unmask_token(probs)
    return update_positions, z_tm1

@torch.no_grad()
def mdlm_adaptive_score_select_update(probs, z_t, t, tm1, i, diffusion_mask, eps=1e-4, score_mask_position=None, select_position=None, unmask_token=None):
    score = score_mask_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    z_tm1 = unmask_token(probs)
    return update_positions, z_tm1

# code for when using MDLMSampler for mdlm strategies
# @torch.no_grad()
# def mdlm_vanilla(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, unmask_token=None):
#     def get_sigmas(t, eps=1e-4):
#         dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
#         print(f'dsigma: {dsigma}')
#         sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
#         return dsigma, sigma
#     # In train for the worst paper, the vanilla inference uses alpha_s and alpha_t. move_chance_tm1 is equal to 1 - alpha_s, move_chance_t is equal to 1 - alpha_t
#     # The weight (move_chance_t - move_chance_tm1) / move_chance_t is equal to the probability (alpha_s - alpha_t) / (1 - alpha_t) to select a token for unmasking
#     _, sigma_t = get_sigmas(t, eps=eps)
#     _, sigma_tm1 = get_sigmas(tm1, eps=eps)
#     # print(f'sigma_t: {sigma_t}')
#     move_chance_t = 1 - torch.exp(-sigma_t)
#     move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
#     prob_unmask_this_step = (move_chance_t - move_chance_tm1) / move_chance_t
#     update_positions = sample_positions_independently(z_t, prob_unmask_this_step)
#     z_tm1 = unmask_token(probs)
#     return update_positions, z_tm1

# @torch.no_grad()
# def mdlm_adaptive_score_select_update(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, score_mask_position=None, select_position=None, unmask_token=None):
#     score = score_mask_position(z_t, probs) * diffusion_mask
#     update_positions = select_position(score)
#     z_tm1 = unmask_token(probs)
#     return update_positions, z_tm1


#################### GIDD sampling strategies ####################

########## tokens stay fixed after unmasking ##########

@torch.no_grad()
def gidd_emulate_mdlm_vanilla(probs, z_t, t, s, i, diffusion_mask, tokenizer, noise_schedule, unmask_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        alpha_t, _ = noise_schedule.get_alpha_betapi(t)
        alpha_s, _ = noise_schedule.get_alpha_betapi(s)
        
        update_positions = sample_positions_independently(z_t, (alpha_s - alpha_t) / (1 - alpha_t)) * (z_t == tokenizer.mask_token_id)
        next_z_t = unmask_token(probs)
    return update_positions, next_z_t # TODO: in last iteration, next_z_t should be selected as max logit, see mdlm implementation

@torch.no_grad()
def gidd_emulate_mdlm_adaptive_score_select_update(probs, z_t, t, s, i, diffusion_mask, tokenizer, score_mask_position, select_position, unmask_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        score = score_mask_position(z_t, probs) * diffusion_mask
        update_positions = select_position(score) * (z_t == tokenizer.mask_token_id)
        next_z_t = unmask_token(probs)
    return update_positions, next_z_t

########## tokens may change even after unmasking i.e. uniform noise is utilized ##########
@torch.no_grad()
def gidd_original(probs, z_t, t, s, i, diffusion_mask, tokenizer, noise_schedule, min_p):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        q_s = noise_schedule.probs_at_t(probs, s)
        q_t = noise_schedule.probs_at_t(probs, t)
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

        alpha_t, beta_pi_t = noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = noise_schedule.get_alpha_betapi(s)

        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

        vocab_size_architecturally = len(tokenizer)
        vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        q_ts = (alpha_ts.unsqueeze(-1) * vz_t + beta_pi_ts_at_zt)

        q_st = q_ts * q_s / q_zt
        
        if min_p > 0.0:
            is_small = (q_st < min_p).float()
            q_st = (1 - is_small) * q_st
            q_st = q_st / q_st.sum(-1, keepdim=True)
        
        is_masked = (z_t == tokenizer.mask_token_id)
        q_st[:, :, tokenizer.mask_token_id] = is_masked.to(dtype=q_st.dtype) * q_st[:, :, tokenizer.mask_token_id]
        q_st = q_st / q_st.sum(-1, keepdim=True)
        
        update_positions = torch.ones_like(z_t, dtype=torch.bool)
        next_z_t = sample_categorical(q_st, end_index=tokenizer.unk_token_id - 1)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_independent_positions_decomposed_update_distribution(probs, z_t, t, s, i, diffusion_mask, tokenizer, noise_schedule, min_p, change_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        q_s = noise_schedule.probs_at_t(probs, s)
        q_t = noise_schedule.probs_at_t(probs, t)
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

        alpha_t, beta_pi_t = noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = noise_schedule.get_alpha_betapi(s)

        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

        vocab_size_architecturally = len(tokenizer)
        vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        q_ts = (alpha_ts.unsqueeze(-1) * vz_t + beta_pi_ts_at_zt)

        q_st = q_ts * q_s / q_zt
        
        if min_p > 0.0:
            is_small = (q_st < min_p).float()
            q_st = (1 - is_small) * q_st
            q_st = q_st / q_st.sum(-1, keepdim=True)
        
        probs_to_change_1m = q_st.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        update_positions = torch.rand_like(probs_to_change_1m) > probs_to_change_1m
        next_z_t = change_token(probs, z_t)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_selected_positions_decomposed_update_distribution(probs, z_t, t, s, i, diffusion_mask, tokenizer, noise_schedule, min_p, score_mask_position, select_position_change, select_position_unmask, change_token, unmask_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        q_s = noise_schedule.probs_at_t(probs, s)
        q_t = noise_schedule.probs_at_t(probs, t)
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

        alpha_t, beta_pi_t = noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = noise_schedule.get_alpha_betapi(s)

        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

        vocab_size_architecturally = len(tokenizer)
        vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        q_ts = (alpha_ts.unsqueeze(-1) * vz_t + beta_pi_ts_at_zt)

        q_st = q_ts * q_s / q_zt
        
        if min_p > 0.0:
            is_small = (q_st < min_p).float()
            q_st = (1 - is_small) * q_st
            q_st = q_st / q_st.sum(-1, keepdim=True)
        
        # Calculate the probability to unmask a token,
        # compare it with the probabilities to change a token.
        # If there exists a token which is more likely to change than it is likely to unmask a token, then change it.
        # Otherwise unmask a token based on the current criteria (position_metric, e.g. max confidence, etc.)
        
        # P(masked s) = P(masked s | unmasked t) * P(unmasked t) + P(masked s | masked t) * P(masked t), P(masked s | unmasked t) = 0
        # -> P(masked s | masked t) = P(masked s) / P(masked t)
        # -> P(unmasked s | masked t) = 1 - P(masked s) / P(masked t)

        # P(masked s | masked t) = P(masked t | masked s) * P(masked s) / P(masked t), P(masked t | masked s) = 1
        # -> P(masked s | masked t) = P(masked s) / P(masked t)

        q_st_at_zt = q_st.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        probs_to_change = 1 - q_st_at_zt
        beta_pi_s_at_mask = beta_pi_s[..., tokenizer.mask_token_id]
        beta_pi_t_at_mask = beta_pi_t[..., tokenizer.mask_token_id]
        probs_to_not_unmask = beta_pi_s_at_mask / beta_pi_t_at_mask
        probs_to_not_unmask = probs_to_not_unmask * (alpha_t.squeeze(-1) / alpha_s.squeeze(-1) + beta_pi_t_at_mask - alpha_t.squeeze(-1) / alpha_s.squeeze(-1) * beta_pi_s_at_mask)
        probs_to_unmask = 1 - probs_to_not_unmask
        probs_to_unmask = probs_to_unmask.unsqueeze(-1).expand_as(probs_to_change)

        probs_to_change = probs_to_change * diffusion_mask
        probs_to_unmask = probs_to_unmask * diffusion_mask

        positions_to_change = (probs_to_change > probs_to_unmask + 1e-6)
        samples_where_change = positions_to_change.any(dim=-1)
        samples_where_unmask = ~samples_where_change

        score_change = probs_to_change * positions_to_change * diffusion_mask
        update_positions_change = select_position_change(score_change)
        next_z_t_change = change_token(probs, z_t)

        score_unmask = score_mask_position(z_t, probs) * (z_t == tokenizer.mask_token_id) * diffusion_mask
        update_positions_unmask = select_position_unmask(score_unmask)
        update_positions_unmask = update_positions_unmask * samples_where_unmask.unsqueeze(-1)
        next_z_t_unmask = unmask_token(probs)

        next_z_t = update_positions_change * next_z_t_change + update_positions_unmask * next_z_t_unmask
        update_positions = update_positions_change | update_positions_unmask
    return update_positions, next_z_t


def gidd_change_based_on_model_confidence_to_change(probs, z_t, t, s, i, diffusion_mask, tokenizer, score_position_for_change, select_position, change_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        score = score_position_for_change(z_t, probs) * diffusion_mask
        update_positions = select_position(score)
        next_z_t = change_token(probs, z_t)
    return update_positions, next_z_t

def gidd_keep_where_confident(probs, z_t, t, s, i, diffusion_mask, tokenizer, score_position_for_keep_where_confident, select_position, change_token):
    if i == 0:
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        score = score_position_for_keep_where_confident(z_t, probs) * diffusion_mask
        update_positions = select_position(score)
        next_z_t = change_token(probs, z_t)
    return update_positions, next_z_t

def gidd_change_low_confidence_positions(probs, z_t, t, s, i, diffusion_mask, tokenizer, score_mask_position, select_position_unmask, unmask_token, change_token):
    if i == 0:
        # print(f'num masks remaining: {(z_t[..., -81:] == tokenizer.mask_token_id).sum(-1)}')
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        confidence_current_token = probs.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        max_confidence = probs.max(-1).values
        low_confidence_positions = (confidence_current_token < max_confidence) * (z_t != tokenizer.mask_token_id) * diffusion_mask.to(dtype=bool)
        samples_where_change = low_confidence_positions.any(dim=-1)
        next_z_t_change = change_token(probs, z_t)

        score_unmask = score_mask_position(z_t, probs) * (z_t == tokenizer.mask_token_id) * diffusion_mask
        update_positions_unmask = select_position_unmask(score_unmask)
        update_positions_unmask = update_positions_unmask * ~samples_where_change.unsqueeze(-1)
        next_z_t_unmask = unmask_token(probs)

        update_positions = low_confidence_positions | update_positions_unmask
        next_z_t = low_confidence_positions * next_z_t_change + update_positions_unmask * next_z_t_unmask
    return update_positions, next_z_t

def gidd_change_lowest_confidence_position(probs, z_t, t, s, i, diffusion_mask, tokenizer, score_mask_position, select_position_unmask, unmask_token, k):
    if i == 0:
        # print(f'num masks remaining: {(z_t[..., -81:] == tokenizer.mask_token_id).sum(-1)}')
        update_positions = (z_t == tokenizer.mask_token_id)
        next_z_t = probs.argmax(-1)
    else:
        confidence_current_token = probs.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        unmasked_positions = (z_t != tokenizer.mask_token_id) * diffusion_mask.to(dtype=bool)
        min_confidence_of_unmasked_indices = torch.topk(confidence_current_token.masked_fill(~unmasked_positions, 1), k=k, largest=False).indices
        update_positions_change = torch.zeros_like(z_t, dtype=torch.bool)
        update_positions_change.scatter_(-1, min_confidence_of_unmasked_indices, True)
        update_positions_change = update_positions_change * unmasked_positions
        next_z_t_change = probs.argmax(-1)
        update_positions_change = update_positions_change * (next_z_t_change != z_t)
        unmask_sample = (update_positions_change.sum(dim=-1) == 0)

        score_unmask = score_mask_position(z_t, probs) * (z_t == tokenizer.mask_token_id) * diffusion_mask
        update_positions_unmask = select_position_unmask(score_unmask)
        update_positions_unmask = update_positions_unmask * unmask_sample.unsqueeze(-1)
        next_z_t_unmask = unmask_token(probs) * update_positions_unmask

        update_positions = update_positions_change | update_positions_unmask
        next_z_t = next_z_t_change * update_positions_change + next_z_t_unmask
    return update_positions, next_z_t

def gidd_flattened(probs:torch.Tensor, z_t, t, s, i, num_denoising_steps, diffusion_mask, max_score, tokenizer):
    # set probs to zero where diffusion_mask is zero
    # flatten probs
    # get the (num_denoising_steps - i) most probable tokens
    # undo the flattening
    # if no token is selected from a position, next_z_t will contain the same token as z_t, reset the max score for this position
    # if exactly one token is selected from a position, next_z_t will contain that, reset the max score for this position
    # if more than one token is selected from a position use the following process to determine the token for next_z_t:
    # - compute a score as (probability of most probable token - probability of token in z_t)
    # - if it is greater than the previous max of this difference, then select the most probable token and update the max score
    # - otherwise keep the token from z_t
    # return the positions to update, next_z_t and the updated max_score
    seq_len = probs.shape[-2]
    vocab_size = probs.shape[-1]
    probs = probs * diffusion_mask.unsqueeze(-1)
    probs_flattened = probs.view((-1, seq_len * vocab_size))
    num_tokens = num_denoising_steps - i
    topk_indices = probs_flattened.topk(num_tokens, dim=-1, sorted=False).indices
    selected_mask = torch.zeros_like(probs_flattened, dtype=torch.bool)
    selected_mask.scatter_(-1, topk_indices, True)
    selected_mask = selected_mask.view(probs.shape)
    update_positions = selected_mask.any(dim=-1)
    has_multiple_selected_tokens = selected_mask.sum(dim=-1) > 1
    largest_probs, largest_probs_indices = probs.max(dim=-1)
    probs_at_z_t = probs.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    score = largest_probs - probs_at_z_t
    score = torch.where(has_multiple_selected_tokens, score, torch.zeros_like(score))
    has_new_max = score > max_score
    max_score = torch.where(has_new_max, score, max_score)
    max_score = torch.where(has_multiple_selected_tokens, max_score, torch.zeros_like(max_score))
    # next_z_t = torch.where(has_new_max, largest_probs_indices, z_t) # alternative
    next_z_t = torch.where(has_new_max, largest_probs_indices, tokenizer.mask_token_id)
    next_z_t = torch.where(update_positions & ~has_multiple_selected_tokens, largest_probs_indices, next_z_t)
    return update_positions, next_z_t, max_score


def gidd_prob_to_recover_data(probs, z_t, t, s, i, diffusion_mask, tokenizer, noise_schedule, min_p):
    # denoising event: p(z_s = x, z_t != x)
    # p(z_s = x) = p(z_t = x) * p(z_s = x | z_t = x) + p(z_t != x) * p(z_s = x | z_t != x)

    # p(z_s = x, z_t != x) = q_t|s(z_t != x | z_s = x) * q_s(z_s = x | x) / q_t(z_t != x | x)
    # = (alpha_s + beta_pi_s_at_zt) * beta_pi_ts_at_zt / beta_pi_t_at_zt # independent of prediction for x

    # p(z_s = x | z_t = x) = q_t|s(z_t = x | z_s = x) * q_s(z_s = x | x) / q_t(z_t = x | x)
    # = (alpha_ts * z_s + beta_pi_ts) * (alpha_s + beta_pi_s) / (alpha_t + beta_pi_t)
    # use x_theta = probs for z_s and take the element at z_t:
    # (alpha_ts * x_theta_at_zt + beta_pi_ts_at_zt) * (alpha_s + beta_pi_s_at_zt) / (alpha_t + beta_pi_t_at_zt)

    alpha_t, beta_pi_t = noise_schedule.get_alpha_betapi(t)
    alpha_s, beta_pi_s = noise_schedule.get_alpha_betapi(s)

    alpha_ts = alpha_t / alpha_s
    beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

    vocab_size_architecturally = len(tokenizer)
    vocab_size_semantically = tokenizer.mask_token_id
    vz_t = F.one_hot(z_t, num_classes=vocab_size_architecturally)
    beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
    beta_pi_s_at_zt = beta_pi_s.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
    beta_pi_t_at_zt = beta_pi_t.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))

    # z_t fixed
    p_zs_x_cond_zt_nx = (alpha_s + beta_pi_s_at_zt) * (beta_pi_ts_at_zt / beta_pi_t_at_zt)
    p_zs_x_cond_zt_x = (alpha_ts * x_theta_at_zt + beta_pi_ts_at_zt) * (alpha_s + beta_pi_s_at_zt) / (alpha_t + beta_pi_t_at_zt)
    x_theta_at_zt = probs.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    p_zt_x = x_theta_at_zt # either from model (x_theta_at_zt) or from recurrence (p_zs_x of previous step)
    p_zt_nx = 1 - p_zt_x
    p_zs_x = p_zt_x * p_zs_x_cond_zt_x + p_zt_nx * p_zs_x_cond_zt_nx

    # z_t based on forward distribution
    p_zs_x_and_zt_x = (alpha_s + beta_pi_s[0]) * (alpha_ts + beta_pi_ts[0])
    p_zs_x_and_zt_nx = (alpha_s + beta_pi_s[0]) * (beta_pi_ts[tokenizer.mask_token_id] + (vocab_size_semantically - 2) * beta_pi_ts[0])
    p_zs_x = p_zs_x_and_zt_x + p_zs_x_and_zt_nx

    # decide whether to update based on p(z_t != x && z_s = x)
    update_positions = torch.rand_like(p_zs_x_and_zt_nx) < p_zs_x_and_zt_nx

    # update selected positions
    next_z_t = sample_token_change_max(probs, z_t)

    # use p_zs_x as stopping criterion
    
    pass



def get_sampling_strategy_class(config, model, noise_schedule, tokenizer, t_eps, min_p):
    if config.strategy == "gidd_prob_to_recover_data":
        return Gidd_prob_to_recover_data(config.p_denoise, model, noise_schedule, tokenizer, config.time_steps, t_eps, config.k, get_self_correction(config))
    elif config.strategy in ["mdlm_vanilla", "mdlm_adaptive_score_select_update"]:
        return Independent_steps(model, None, tokenizer, config.time_steps, t_eps, get_sampling_strategy(config, tokenizer, None, min_p), get_self_correction(config))
    else:
        return Independent_steps(model, noise_schedule, tokenizer, config.time_steps, t_eps, get_sampling_strategy(config, tokenizer, noise_schedule, min_p), get_self_correction(config))

class SamplingStrategy(nn.Module):
    def __init__(self, model, noise_schedule, tokenizer, time_steps, t_eps):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.time_steps = time_steps
        self.t_eps = t_eps

    @torch.no_grad()
    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        self.initial_z_t = initial_z_t.to(device, non_blocking=True)
        self.diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        self.solution = solution.to(device, non_blocking=True)
        self.puzzle_conditioning = puzzle_conditioning
        self.num_denoising_steps = num_denoising_steps
        self.num_self_correction_steps = num_self_correction_steps
        self.max_length = max_length
        self.device = device
        self.ts = torch.linspace(0, 1, self.num_denoising_steps + 1, device=device).unsqueeze(-1)
        self.ts = (1 - 2 * self.t_eps) * self.ts + self.t_eps
        # State which is updated throughout sampling
        self.z_t = self.initial_z_t.clone()
        self.p_zs_x = torch.zeros_like(self.initial_z_t, dtype=torch.float, device=device)
        self.p_zs_x_mean_EMA = torch.zeros((self.initial_z_t.shape[0]), dtype=torch.float, device=device)
        self.is_fully_denoised = torch.zeros(self.initial_z_t.shape[0], dtype=torch.bool, device=device)
        self.has_self_corrected = False

    @abstractmethod
    @torch.no_grad()
    def stopping_criterion(self, i, generation_info_handler):
        raise NotImplementedError
    
    @abstractmethod
    @torch.no_grad()
    def step(self, i, generation_info_handler, blocked_logits_mask=None):
        raise NotImplementedError
    
    def infer_time_step(self):
        p_zs_x_mean = mean_over_diffusion_positions(self.p_zs_x, self.diffusion_mask)
        p_zs_x_mean_EMA_weight = 0.8
        self.p_zs_x_mean_EMA = p_zs_x_mean_EMA_weight * self.p_zs_x_mean_EMA + (1 - p_zs_x_mean_EMA_weight) * p_zs_x_mean
        current_ts = 1 - self.p_zs_x_mean_EMA.unsqueeze(-1).expand(-1, self.ts.shape[0])
        current_ts = (1 - 2 * self.t_eps) * current_ts + self.t_eps
        current_ts_indices = (current_ts - self.ts.squeeze(-1)).abs().argmin(-1)
        t = self.ts[current_ts_indices].squeeze(-1)
        s = self.ts[torch.maximum(current_ts_indices - 1, torch.zeros_like(current_ts_indices))].squeeze(-1)
        return t, s
    
    def get_state(self):
        return {
            'z_t': self.z_t.clone(),
            'p_zs_x': self.p_zs_x.clone(),
            'p_zs_x_mean_EMA': self.p_zs_x_mean_EMA.clone(),
            'is_fully_denoised': self.is_fully_denoised.clone(),
            'has_self_corrected': self.has_self_corrected
        }
    
    def set_state(self, beam):
        self.z_t = beam['z_t'].clone()
        self.p_zs_x = beam['p_zs_x'].clone()
        self.p_zs_x_mean_EMA = beam['p_zs_x_mean_EMA'].clone()
        self.is_fully_denoised = beam['is_fully_denoised'].clone()
        self.has_self_corrected = beam['has_self_corrected']

    def branch_step(self, beam, branching_factor, generation_info_handler=None):
        new_beams = []
        blocked_logits_mask = torch.zeros((beam['z_t'].shape[0], beam['z_t'].shape[1], self.tokenizer.mask_token_id), dtype=torch.bool, device=self.device)
        original_z_t = beam['z_t']
        for _ in range(branching_factor):
            self.set_state(beam)
            self.step(beam['step'], generation_info_handler=generation_info_handler, blocked_logits_mask=blocked_logits_mask)
            new_beam = self.get_state()
            new_beams.append(new_beam)
            # update blocked_logits_mask
            new_z_t = new_beam['z_t']
            updated_positions = (original_z_t != new_z_t)
            for sample_idx in range(original_z_t.shape[0]):
                for position_idx in range(original_z_t.shape[1]):
                    if updated_positions[sample_idx, position_idx]:
                        new_value = new_z_t[sample_idx, position_idx].item()
                        blocked_logits_mask[sample_idx, position_idx, new_value] = 1
        return new_beams

    def beam_score(self, beam, score_time='t_zero', score_method='avg', generation_info_handler=None): # shape of z_t: (batch_size=1, seq_len)
        if generation_info_handler is not None:
            generation_info_handler.beam_search_forward_calls_step()
        self.set_state(beam)
        if score_time == 'inferred':
            t, _ = self.infer_time_step()
        else:
            t = self.ts[0]
        logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
        logits[..., self.tokenizer.mask_token_id:] = -1e6
        probs = logits.softmax(-1)
        p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
        if score_method == 'avg':
            p_zt_x_mean = mean_over_diffusion_positions(p_zt_x, self.diffusion_mask)
            return p_zt_x_mean.squeeze(0).item()
        elif score_method == 'avg_unmasked':
            is_unmasked = torch.logical_and(self.diffusion_mask.to(dtype=bool), self.z_t != self.tokenizer.mask_token_id)
            p_zt_x_unmasked = torch.where(is_unmasked, p_zt_x, 0)
            num_unmasked = torch.sum(is_unmasked).clamp(min=1)
            p_zt_x_mean_unmasked = torch.sum(p_zt_x_unmasked) / num_unmasked
            return p_zt_x_mean_unmasked.squeeze(0).item()
        elif score_method == 'avg_all_using_confidence_for_masked':
            is_masked = torch.logical_and(self.diffusion_mask.to(dtype=bool), self.z_t == self.tokenizer.mask_token_id)
            is_unmasked = torch.logical_and(self.diffusion_mask.to(dtype=bool), self.z_t != self.tokenizer.mask_token_id)
            confidence_for_masked = torch.where(is_masked, probs.max(-1).values, 0)
            p_zt_x_unmasked = torch.where(is_unmasked, p_zt_x, 0)
            return ((torch.sum(confidence_for_masked) + torch.sum(p_zt_x_unmasked)) / torch.sum(self.diffusion_mask)).squeeze(0).item()
        elif score_method == 'min':
            p_zt_x = torch.where(torch.logical_and(self.diffusion_mask.to(dtype=bool), self.z_t != self.tokenizer.mask_token_id), p_zt_x, 1)
            min_p_zt_x = torch.min(p_zt_x, dim=-1).values
            return min_p_zt_x.squeeze(0).item()
     
    def beam_score_final(self, beam, generation_info_handler=None): # shape of z_t: (batch_size=1, seq_len)
        if generation_info_handler is not None:
            generation_info_handler.beam_search_forward_calls_step()
        z_t = beam['z_t']
        logits = self.model(z_t, self.ts[0], puzzle_conditioning=self.puzzle_conditioning)
        logits[..., self.tokenizer.mask_token_id:] = -1e6
        probs = logits.softmax(-1)
        p_zt_x = probs.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        p_zt_x = torch.where(self.diffusion_mask.to(dtype=bool), p_zt_x, 1)
        min_p_zt_x = torch.min(p_zt_x, dim=-1).values
        return min_p_zt_x.squeeze(0).item()
    
    def beam_score_final_accepted(self, beam, generation_info_handler=None):
        score = self.beam_score_final(beam, generation_info_handler=generation_info_handler)
        # if score >= 0.9995:
        #     print(f'Final beam min p_zt_x: {score} ACCEPTED')
        return score >= 0.9995


class Independent_steps(SamplingStrategy):
    def __init__(self, model, noise_schedule, tokenizer, time_steps, t_eps, sampling_strategy, self_correction=None):
        super().__init__(model, noise_schedule, tokenizer, time_steps, t_eps)
        self.sampling_strategy = sampling_strategy
        self.self_correction = self_correction

    @torch.no_grad()
    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
    
    def stopping_criterion(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            stopping_criteria_met = self.is_fully_denoised.all() and self.self_correction is None
            if generation_info_handler is not None and stopping_criteria_met:
                    generation_info_handler.batch_step_history(self.z_t)
                    generation_info_handler.batch_step_marginals_final(self.z_t)
            return stopping_criteria_met
        else:
            return self.self_correction is None or self.has_self_corrected
    
    def step(self, i, generation_info_handler, blocked_logits_mask=None):
        if i < self.num_denoising_steps:
            self.is_fully_denoised = (self.z_t[:, -self.max_length:] != self.tokenizer.mask_token_id).all(dim=1)
            if self.is_fully_denoised.all():
                return
            if generation_info_handler is not None:
                generation_info_handler.batch_step_forward_calls(self.is_fully_denoised)
                generation_info_handler.beam_search_forward_calls_step()
            if self.time_steps == 'fixed':
                t = self.ts[self.num_denoising_steps - 1 - i]
                s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
            elif self.time_steps == 'inferred':
                t, s = self.infer_time_step()
            logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)
            if blocked_logits_mask is not None:
                probs[..., :self.tokenizer.mask_token_id] = torch.where(blocked_logits_mask, 0, probs[..., :self.tokenizer.mask_token_id])

            update_positions, next_z_t = self.sampling_strategy(probs, self.z_t, t, s, self.num_denoising_steps - 1 - i, self.diffusion_mask)

            update_positions = update_positions * self.diffusion_mask
            update_positions = update_positions & ~self.is_fully_denoised.unsqueeze(-1)
            
            if generation_info_handler is not None:
                generation_info_handler.batch_step_history(self.z_t)
                generation_info_handler.batch_step_logits(logits[..., :self.tokenizer.mask_token_id])
                generation_info_handler.batch_step_marginals(self.z_t, probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1), s, self.tokenizer.mask_token_id)

            self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)

            self.p_zs_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)

            if generation_info_handler is not None:
                if i == self.num_denoising_steps - 1:
                    generation_info_handler.batch_step_history(self.z_t)
                    logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    generation_info_handler.batch_step_marginals(self.z_t, p_zt_x, s, self.tokenizer.mask_token_id)
                generation_info_handler.batch_step_change_events(i, self.z_t, probs, self.tokenizer.mask_token_id)

                if getattr(generation_info_handler, "collect_confidence_t_0", False):
                    logits = self.model(self.z_t, self.ts[0], puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    generation_info_handler.batch_step_confidence_t_0(probs)

                correct_samples = (self.z_t == generation_info_handler.batch_solution).all(dim=-1)
                if correct_samples.any():
                    logits = self.model(self.z_t, self.ts[0], puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    p_zt_x = torch.where(self.diffusion_mask.to(dtype=bool), p_zt_x, 1)
                    
                    # print p_zt_x for the correct samples
                    # print(f'step {i}')
                    # print(f'num correct samples: {correct_samples.sum().item()}')
                    min_p_zt_x_correct = torch.min(p_zt_x[correct_samples, -81:])
                    # print(f'min p_zt_x:{min_p_zt_x_correct}')
                    # correct_samples_indices = torch.nonzero(correct_samples).squeeze(-1).tolist()
                    # print(f'correct samples indices: {correct_samples_indices}')
                    # min_p_zt_x_index = torch.argmin(p_zt_x[correct_samples, -81:])
                    # print(f'index of min p_zt_x: token {min_p_zt_x_index % 81} in sample {correct_samples_indices[min_p_zt_x_index // 81]}')
                    
                    # print largest min p_zt_x for incorrect samples
                    # incorrect_samples = ~correct_samples
                    # if incorrect_samples.any():
                    #     max_min_p_zt_x = torch.max(torch.min(p_zt_x[incorrect_samples, -81:], dim=-1).values))
                    #     if max_min_p_zt_x > 0.9:
                    #         print(f'large min p_zt_x for incorrect samples: {max_min_p_zt_x}, min p_zt_x for correct samples: {min_p_zt_x_correct}, separable: {min_p_zt_x_correct > max_min_p_zt_x}')

        elif self.self_correction is not None and not self.has_self_corrected:
            # TODO: change self-correction code to update one step at a time, such that history can be collected easily
            self.z_t, _ = self.self_correction(self.model, self.tokenizer, self.diffusion_mask, self.z_t, self.ts[0].item(), max_num_denoising_steps=self.num_self_correction_steps)
            self.has_self_corrected = True
            if generation_info_handler is not None:
                generation_info_handler.batch_step_history(self.z_t)

class Gidd_flattened(SamplingStrategy):
    def __init__(self, model, noise_schedule, tokenizer, t_eps, sampling_strategy, self_correction=None):
        super().__init__(model, noise_schedule, tokenizer, t_eps)
        self.sampling_strategy = sampling_strategy
        self.self_correction = self_correction

    @torch.no_grad()
    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        self.ts = torch.linspace(0, 1, self.num_denoising_steps + 1, device=device).unsqueeze(-1)
        self.ts = (1 - 2 * self.t_eps) * self.ts + self.t_eps
        self.z_t = self.initial_z_t.clone()
        self.max_score = torch.zeros_like(self.z_t, dtype=torch.float, device=device)
        self.is_fully_denoised = torch.zeros(self.initial_z_t.shape[0], dtype=torch.bool, device=device)
        self.has_self_corrected = False
    
    def stopping_criterion(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            return self.is_fully_denoised.all() and self.self_correction is None
        else:
            return self.self_correction is None or self.has_self_corrected
        
    def step(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            self.is_fully_denoised = (self.z_t[:, -self.max_length:] != self.tokenizer.mask_token_id).all(dim=1)
            if self.is_fully_denoised.all():
                return
            t = self.ts[self.num_denoising_steps - 1 - i]
            s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
            logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)

            update_positions, next_z_t, self.max_score = self.sampling_strategy(probs, self.z_t, t, s, self.num_denoising_steps - 1 - i, self.num_denoising_steps, self.diffusion_mask, self.max_score)

            update_positions = update_positions * self.diffusion_mask
            update_positions = update_positions & ~self.is_fully_denoised.unsqueeze(-1)
            self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)
        elif self.self_correction is not None and not self.has_self_corrected:
            # TODO: change self-correction code to update one step at a time, such that history can be collected easily
            self.z_t, _ = self.self_correction(self.model, self.tokenizer, self.diffusion_mask, self.z_t, self.ts[0].item(), max_num_denoising_steps=self.num_self_correction_steps)
            self.has_self_corrected = True

class Gidd_prob_to_recover_data(SamplingStrategy):
    def __init__(self, config, model, noise_schedule, tokenizer, time_steps, t_eps, k, self_correction=None):
        super().__init__(model, noise_schedule, tokenizer, time_steps, t_eps)
        self.config = config
        self.k = k
        self.self_correction = self_correction

    @torch.no_grad()
    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        # State which remains constant throughout sampling
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        self.not_mask_token_id_tensor = torch.ones((1, 1), dtype=initial_z_t.dtype, device=device) * self.noise_schedule.not_mask_id
        self.mask_token_id_tensor = torch.ones((1, 1), dtype=initial_z_t.dtype, device=device) * self.tokenizer.mask_token_id

    def stopping_criterion(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            stopping_criteria_met = self.is_fully_denoised.all() and self.self_correction is None
            if generation_info_handler is not None and stopping_criteria_met:
                generation_info_handler.batch_step_history(self.z_t)
            return stopping_criteria_met
        else:
            return self.self_correction is None or self.has_self_corrected

    def step(self, i, generation_info_handler, blocked_logits_mask=None):
        if i < self.num_denoising_steps:
            if self.is_fully_denoised.all():
                return
            if generation_info_handler is not None:
                generation_info_handler.batch_step_forward_calls(self.is_fully_denoised)
                generation_info_handler.beam_search_forward_calls_step()
            if self.time_steps == 'fixed':
                t = self.ts[self.num_denoising_steps - 1 - i]
                s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
            elif self.time_steps == 'inferred':
                t, s = self.infer_time_step()
            logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)
            if blocked_logits_mask is not None:
                probs[..., :self.tokenizer.mask_token_id] = torch.where(blocked_logits_mask, 0, probs[..., :self.tokenizer.mask_token_id])
            
            if i == self.num_denoising_steps - 1:
                update_positions = (self.z_t == self.tokenizer.mask_token_id) * self.diffusion_mask
                next_z_t = probs.argmax(-1)
                p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                uniform_noise_positions = torch.zeros_like(p_zt_x, dtype=torch.bool)
                next_z_t_noise = next_z_t
            else:
                alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
                alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

                alpha_ts = alpha_t / alpha_s
                beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

                vocab_size_architecturally = len(self.tokenizer)
                vocab_size_semantically = self.tokenizer.mask_token_id
                vz_t = F.one_hot(self.z_t, num_classes=vocab_size_architecturally)
                beta_pi_s_at_zt = beta_pi_s.unsqueeze(1).expand_as(vz_t).gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                beta_pi_t_at_zt = beta_pi_t.unsqueeze(1).expand_as(vz_t).gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                beta_pi_s_at_not_m = beta_pi_s.gather(-1, self.not_mask_token_id_tensor)
                beta_pi_t_at_not_m = beta_pi_t.gather(-1, self.not_mask_token_id_tensor)
                beta_pi_ts_at_not_m = beta_pi_ts.gather(-1, self.not_mask_token_id_tensor)
                beta_pi_s_at_m = beta_pi_s.gather(-1, self.mask_token_id_tensor)
                beta_pi_t_at_m = beta_pi_t.gather(-1, self.mask_token_id_tensor)
                beta_pi_ts_at_m = beta_pi_ts.gather(-1, self.mask_token_id_tensor)
                
                # denoising event: p(z_s = x, z_t != x)
                # p(z_s = x) = p(z_t = x) * p(z_s = x | z_t = x) + p(z_t != x) * p(z_s = x | z_t != x)

                # z_t fixed
                if self.config.oracle == "perfect":
                    p_zt_x = (self.z_t == self.solution).to(dtype=torch.float) # using perfect oracle for p_zt_x
                elif self.config.oracle == "model":
                    p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1) # using the model predictions
                elif self.config.oracle == "recurrence":
                    is_mask_token = self.z_t == self.tokenizer.mask_token_id
                    p_zt_x = torch.where(is_mask_token, 0, self.p_zs_x)
                elif self.config.oracle == "model_and_recurrence":
                    p_zt_x_model = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    is_mask_token = self.z_t == self.tokenizer.mask_token_id
                    p_zt_x_recurrence = torch.where(is_mask_token, 0, self.p_zs_x)
                    weight = 0.5
                    p_zt_x = weight * p_zt_x_model + (1 - weight) * p_zt_x_recurrence
                elif self.config.oracle == "model_EMA":
                    p_zt_x_model = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    weight = 0.1
                    p_zt_x = weight * p_zt_x_model + (1 - weight) * self.p_zs_x

                p_zs_x_and_zt_x = p_zt_x * (alpha_s + beta_pi_s_at_not_m) * (alpha_ts + beta_pi_ts_at_not_m) / (alpha_t + beta_pi_t_at_not_m)
                p_zs_x_and_zt_nx = (1 - p_zt_x) * (alpha_s + beta_pi_s_at_not_m) * beta_pi_ts_at_zt / beta_pi_t_at_zt

                # p_zs_nx_and_zt_nx = (1 - p_zt_x) * (1 - (alpha_s + beta_pi_s_at_not_m) * beta_pi_ts_at_zt / beta_pi_t_at_zt)
                p_zs_u_and_zt_m = (vocab_size_semantically - 2) * beta_pi_s_at_not_m * beta_pi_ts_at_m / beta_pi_t_at_m
                # print(f"{i}: p_zs_u_and_zt_m: {p_zs_u_and_zt_m.max()}, {p_zs_u_and_zt_m.min()}") # ~1-2%
                
                # use p_zt_x as stopping criterion # TODO: Investigate effect of current stopping criterion and try others
                p_zt_x_mean = mean_over_diffusion_positions(p_zt_x, self.diffusion_mask)
                denoised = p_zt_x > 0.9
                denoised = denoised | ~self.diffusion_mask.bool() | self.is_fully_denoised.unsqueeze(-1)
                self.is_fully_denoised = denoised.all(dim=-1)

                # z_t based on forward distribution # issue: p_zs_x_and_zt_nx is small, not enough time for all positions to unmask (even with more steps, tried 300)
                # p_zs_x_and_zt_x = (alpha_s + beta_pi_s_at_not_m) * (alpha_ts + beta_pi_ts_at_not_m)
                # p_zs_x_and_zt_nx = (alpha_s + beta_pi_s_at_not_m) * (beta_pi_ts_at_m + (vocab_size_semantically - 2) * beta_pi_ts_at_not_m).expand_as(self.z_t)
                # p_zs_x = p_zs_x_and_zt_x + p_zs_x_and_zt_nx

                # decide whether to update based on p(z_s = x && z_t != x)
                uniform_noise_positions = torch.zeros_like(p_zt_x, dtype=torch.bool)
                if self.config.position_sampling == "independent":
                    dice_roll = torch.rand_like(p_zt_x)
                    if self.config.position_metric == "p_denoise":
                        update_positions = dice_roll < p_zs_x_and_zt_nx
                        # uniform_noise_positions = (dice_roll >= p_zs_x_and_zt_nx) & (dice_roll < p_zs_x_and_zt_nx + p_zs_u_and_zt_m)
                        uniform_noise_positions = (dice_roll >= p_zs_x_and_zt_nx) & (dice_roll < p_zs_x_and_zt_nx + p_zs_u_and_zt_m) & (self.z_t == self.tokenizer.mask_token_id)
                    elif self.config.position_metric == "confident_and_p_denoise":
                        confident_and_p_denoise = probs.max(-1).values * p_zs_x_and_zt_nx
                        update_positions = dice_roll < confident_and_p_denoise
                    elif self.config.position_metric == "confident_and_noisy":
                        confident_and_noisy = probs.max(-1).values * (1 - p_zt_x)
                        update_positions = dice_roll < confident_and_noisy
                    elif self.config.position_metric == "margin_and_noisy":
                        top2 = torch.topk(probs, 2, dim=-1)
                        margin = top2.values[..., 0] - top2.values[..., 1]
                        margin_and_noisy = margin * (1 - p_zt_x)
                        update_positions = dice_roll < margin_and_noisy
                    elif self.config.position_metric == "noisy":
                        noisy = 1 - p_zt_x
                        update_positions = dice_roll < noisy
                    elif self.config.position_metric == "confident":
                        confident = probs.max(-1).values
                        update_positions = dice_roll < confident
                    elif self.config.position_metric == "margin":
                        top2 = torch.topk(probs, 2, dim=-1)
                        margin = top2.values[..., 0] - top2.values[..., 1]
                        update_positions = dice_roll < margin
                elif self.config.position_sampling == "top_k":
                    # updatable_positions = self.diffusion_mask
                    updatable_positions = self.diffusion_mask & ~denoised
                    ks = torch.ones(updatable_positions.shape[0], device=updatable_positions.device) * self.k
                    # ks = torch.where(p_zt_x_mean > 0.3, 4, ks)
                    # ks = torch.where(p_zt_x_mean > 0.6, 2, ks)
                    # ks = torch.where(p_zt_x_mean > 0.8, 1, ks)#seq_len
                    if self.config.position_metric == "p_denoise":
                        metric = p_zs_x_and_zt_nx * updatable_positions
                        # threshold = 0.9
                        # ks = torch.clamp((metric > threshold).sum(-1), min=1)
                        update_positions = sample_position_top_variable_k(metric, ks) # select positions with top-k probabilities to have a denoising event
                    elif self.config.position_metric == "confident_and_p_denoise":
                        metric = probs.max(-1).values * p_zs_x_and_zt_nx * updatable_positions
                        # threshold = 0.1
                        # ks = torch.clamp((metric > threshold).sum(-1), min=1)
                        update_positions = sample_position_top_variable_k(metric, ks)
                    elif self.config.position_metric == "confident_and_noisy":
                        metric = probs.max(-1).values * (1 - p_zt_x) * updatable_positions
                        # threshold = 1.0
                        # ks = torch.clamp((metric > threshold).sum(-1), min=1)
                        update_positions = sample_position_top_variable_k(metric, ks)
                    elif self.config.position_metric == "margin_and_noisy":
                        top2 = torch.topk(probs, 2, dim=-1)
                        margin = top2.values[..., 0] - top2.values[..., 1]
                        metric = margin * (1 - p_zt_x) * updatable_positions
                        # max_metric_per_sample = torch.max(metric, dim=-1).values
                        # threshold = 0.98 * max_metric_per_sample
                        # ks = (metric >= threshold.unsqueeze(-1)).sum(-1)
                        # ks = torch.clamp(ks, min=1, max=10)
                        # if i % 10 == 0:
                        #     print(f"i: {i}, ks: {ks}")
                        update_positions = sample_position_top_variable_k(metric, ks)

                # update selected positions
                if self.config.token_sampling == "categorical":
                    next_z_t = sample_categorical(probs, end_index=self.tokenizer.unk_token_id - 1) # sample categorically from predictions
                elif self.config.token_sampling == "change_max":
                    next_z_t = sample_token_change_max(probs, self.z_t) # force change
                elif self.config.token_sampling == "max":
                    next_z_t = sample_token_MDM_max(probs) # don't force change
                
                # update positions selected for uniform noise
                if self.config.uniform_noise == "none":
                    next_z_t_noise = self.z_t
                elif self.config.uniform_noise == "noise":
                    next_z_t_noise = torch.randint(0, self.tokenizer.mask_token_id, self.z_t.shape, device=self.device)
                elif self.config.uniform_noise == "model":
                    next_z_t_noise = next_z_t
                
                update_positions = update_positions * self.diffusion_mask
                uniform_noise_positions = uniform_noise_positions * self.diffusion_mask
                # update_positions = update_positions & ~self.is_fully_denoised.unsqueeze(-1) # only prevent updating if entire sample is currently believed to be denoised
                # uniform_noise_positions = uniform_noise_positions & ~self.is_fully_denoised.unsqueeze(-1) # only prevent updating if entire sample is currently believed to be denoised
                update_positions = update_positions & ~denoised # don't update positions that are currently believed to be denoised
                uniform_noise_positions = uniform_noise_positions & ~denoised # don't update positions that are currently believed to be denoised
                # print(f"{i}: num denoised positions: {(denoised & self.diffusion_mask).sum()}")
                
                if self.config.oracle == "model_EMA":
                    self.p_zs_x = torch.where(update_positions.bool(), probs.gather(-1, next_z_t.unsqueeze(-1)).squeeze(-1), p_zt_x)
                else:
                    self.p_zs_x = p_zs_x_and_zt_x + p_zs_x_and_zt_nx
                    if self.config.oracle in ["recurrence", "model_and_recurrence"]:
                        self.p_zs_x = torch.where(update_positions.bool(), probs.gather(-1, next_z_t.unsqueeze(-1)).squeeze(-1), self.p_zs_x)
                    
            if generation_info_handler is not None:
                generation_info_handler.batch_step_history(self.z_t)
                generation_info_handler.batch_step_logits(logits[..., :self.tokenizer.mask_token_id])
                generation_info_handler.batch_step_marginals(self.z_t, p_zt_x, s, self.tokenizer.mask_token_id)

            self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)
            self.z_t = torch.where(uniform_noise_positions.bool(), next_z_t_noise, self.z_t)

            # print(f"{i}: num update positions: {update_positions.sum()}, num uniform noise positions: {uniform_noise_positions.sum()}")

            if generation_info_handler is not None:
                if i == self.num_denoising_steps - 1:
                    generation_info_handler.batch_step_history(self.z_t)
                    logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    generation_info_handler.batch_step_marginals(self.z_t, p_zt_x, s, self.tokenizer.mask_token_id)
                else:#TODO: why not in last step?
                    generation_info_handler.batch_step_change_events(i, self.z_t, probs, self.tokenizer.mask_token_id)
                
                if getattr(generation_info_handler, "collect_confidence_t_0", False):
                    logits = self.model(self.z_t, self.ts[0], puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    generation_info_handler.batch_step_confidence_t_0(probs)

                correct_samples = (self.z_t == generation_info_handler.batch_solution).all(dim=-1)
                if correct_samples.any():
                    logits = self.model(self.z_t, self.ts[0], puzzle_conditioning=self.puzzle_conditioning)
                    logits[..., self.tokenizer.mask_token_id:] = -1e6
                    probs = logits.softmax(-1)
                    p_zt_x = probs.gather(-1, self.z_t.unsqueeze(-1)).squeeze(-1)
                    p_zt_x = torch.where(self.diffusion_mask.to(dtype=bool), p_zt_x, 1)
                    
                    # print p_zt_x for the correct samples
                    # print(f'step {i}')
                    # print(f'num correct samples: {correct_samples.sum().item()}')
                    min_p_zt_x_correct = torch.min(p_zt_x[correct_samples, -81:])
                    # print(f'min p_zt_x:{min_p_zt_x_correct}')
                    # correct_samples_indices = torch.nonzero(correct_samples).squeeze(-1).tolist()
                    # print(f'correct samples indices: {correct_samples_indices}')
                    # min_p_zt_x_index = torch.argmin(p_zt_x[correct_samples, -81:])
                    # print(f'index of min p_zt_x: token {min_p_zt_x_index % 81} in sample {correct_samples_indices[min_p_zt_x_index // 81]}')
                    
                    # print largest min p_zt_x for incorrect samples
                    # incorrect_samples = ~correct_samples
                    # if incorrect_samples.any():
                    #     max_min_p_zt_x = torch.max(torch.min(p_zt_x[incorrect_samples, -81:], dim=-1).values)
                    #     # print(f'largest min p_zt_x for incorrect samples: {max_min_p_zt_x}')
                    #     if max_min_p_zt_x > 0.9:
                    #         print(f'large min p_zt_x for incorrect samples: {max_min_p_zt_x}, min p_zt_x for correct samples: {min_p_zt_x_correct}, separable: {min_p_zt_x_correct > max_min_p_zt_x}')
        
        elif self.self_correction is not None and not self.has_self_corrected:
            # TODO: change self-correction code to update one step at a time, such that history can be collected easily
            self.z_t, _ = self.self_correction(self.model, self.tokenizer, self.diffusion_mask, self.z_t, self.ts[0].item(), max_num_denoising_steps=self.num_self_correction_steps)
            self.has_self_corrected = True
            if generation_info_handler is not None:
                generation_info_handler.batch_step_history(self.z_t)
