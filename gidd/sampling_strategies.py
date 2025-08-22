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
    sample_token_change_max,
    sample_token_change_categorical,
    sample_token_MDM_max,
    sample_token_MDM_categorical,
    sample_categorical,
)
from gidd.self_correction_strategies import get_self_correction

def get_score_mask_position(config, tokenizer):
    match config.sampling.score_mask_position:
        case "MDM_max":
            return partial(score_mask_position_max, tokenizer=tokenizer)
        case "MDM_margin":
            return partial(score_mask_position_margin, tokenizer=tokenizer)

def get_score_position_for_change(config):
    match config.sampling.score_position_for_change:
        case "change_max":
            return partial(score_position_for_change_max)
        case "change_margin":
            return partial(score_position_for_change_margin)

def get_score_position_for_keep_where_confident(config):
    match config.sampling.score_position_for_change:
        case "change_max":
            return partial(score_position_for_keep_where_confident_max)
        case "change_margin":
            return partial(score_position_for_keep_where_confident_margin)

def get_select_position(config, arg_name='select_position'):
    match arg_name:
        case 'select_position':
            arg = config.sampling.select_position
        case 'select_position_change':
            arg = config.sampling.select_position_change
        case 'select_position_unmask':
            arg = config.sampling.select_position_unmask
        case _:
            raise ValueError(f"Unknown argument name: {arg_name}")

    match arg:
        case "all":
            return all_positions # == where metric is not zero
        case "independent":
            return sample_positions_independently
        case "top_k_gumbel":
            return partial(sample_position_top_k_gumbel, k=config.sampling.k, gumbel_noise_coefficient=config.sampling.gumbel_noise_coefficient)

def get_unmask_token(config, tokenizer):
    match config.sampling.unmask_token:
        case "MDM_max":
            return partial(sample_token_MDM_max)
        case "MDM_categorical":
            return partial(sample_token_MDM_categorical, tokenizer=tokenizer)

def get_change_token(config, tokenizer):
    match config.sampling.change_token:
        case "change_max":
            return partial(sample_token_change_max)
        case "change_categorical":
            return partial(sample_token_change_categorical, tokenizer=tokenizer)

def get_sampling_strategy(config, tokenizer, noise_schedule=None, min_p=None):
    match config.sampling.strategy:
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
                           score_position_for_change=get_score_position_for_change(config),
                           select_position=get_select_position(config),
                           change_token=get_change_token(config, tokenizer),)
        case "gidd_keep_where_confident":
            return partial(gidd_keep_where_confident,
                           score_position_for_keep_where_confident=get_score_position_for_keep_where_confident(config),
                           select_position=get_select_position(config),
                           change_token=get_change_token(config, tokenizer),)
        case "gidd_flattened":
            return partial(gidd_flattened,
                           tokenizer=tokenizer,)

#################### MDLM sampling strategies ####################
@torch.no_grad()
def mdlm_vanilla(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, unmask_token=None):
    def get_sigmas(t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma
    # In train for the worst paper, the vanilla inference uses alpha_s and alpha_t. move_chance_tm1 is equal to 1 - alpha_s, move_chance_t is equal to 1 - alpha_t
    # The weight (move_chance_t - move_chance_tm1) / move_chance_t is equal to the probability (alpha_s - alpha_t) / (1 - alpha_t) to select a token for unmasking
    _, sigma_t = get_sigmas(t, eps=eps)
    _, sigma_tm1 = get_sigmas(tm1, eps=eps)
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
    prob_unmask_this_step = (move_chance_t - move_chance_tm1) / move_chance_t
    update_positions = sample_positions_independently(z_t, prob_unmask_this_step)
    z_tm1 = unmask_token(probs)
    return update_positions, z_tm1

@torch.no_grad()
def mdlm_adaptive_score_select_update(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, score_mask_position=None, select_position=None, unmask_token=None):
    score = score_mask_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    z_tm1 = unmask_token(probs)
    return update_positions, z_tm1


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
        q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

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
        # if i == 76 or i == 75:
        #     print(f"q_st: {q_st[0, 81, :10]}")
        #     print(f"q_ts: {q_ts[0, 81, :10]}")
        #     print(f"q_s: {q_s[0, 81, :10]}")
        #     print(f"q_zt: {q_zt[0, 81].item()}")
        #     print(f"z_t: {z_t[0, 81].item()}")
        #     print(f"next_z_t: {next_z_t[0, 81].item()}")
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
        q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

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
        q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

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

        if positions_to_change.any():
            score = probs_to_change * positions_to_change.to(dtype=probs_to_change.dtype)
            score = score * diffusion_mask
            update_positions = select_position_change(score)
            next_z_t = change_token(probs, z_t)
        else:
            score = score_mask_position(z_t, probs)
            score = score * diffusion_mask * (z_t == tokenizer.mask_token_id).to(dtype=score.dtype)
            update_positions = select_position_unmask(score)
            next_z_t = unmask_token(probs)
    return update_positions, next_z_t


def gidd_change_based_on_model_confidence_to_change(probs, z_t, t, s, i, diffusion_mask, score_position_for_change, select_position, change_token):
    score = score_position_for_change(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    next_z_t = change_token(probs, z_t)
    return update_positions, next_z_t

def gidd_keep_where_confident(probs, z_t, t, s, i, diffusion_mask, score_position_for_keep_where_confident, select_position, change_token):
    score = score_position_for_keep_where_confident(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    next_z_t = change_token(probs, z_t)
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
    # if num_tokens == 2:
    #     # next_z_t_for_print = torch.where(diffusion_mask.bool(), next_z_t, 9)
    #     print(f"next_z_t: {next_z_t[0, 81:]}")
    #     # print(f"next_z_t for print: {next_z_t_for_print[0, 81:]}")
    #     torch.set_printoptions(threshold=100_000)
    #     print(f"selected_mask: {selected_mask[0, 81:, :9].to(dtype=int)}")
    #     torch.set_printoptions(profile="default")
    #     print(f"largest_probs: {largest_probs[0, 81:]}")
    #     print(f"num update tokens: {update_positions[0, 81:].sum().item()}")
    #     print(f"num unmasked tokens: {(next_z_t[0, 81:] != tokenizer.mask_token_id).sum().item()}")
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
    match config.sampling.strategy:
        case "gidd_prob_to_recover_data":
            return Gidd_prob_to_recover_data(config.sampling.p_denoise, model, noise_schedule, tokenizer, t_eps)
        case "gidd_flattened":
            return Gidd_flattened(model, noise_schedule, tokenizer, t_eps, get_sampling_strategy(config, tokenizer, noise_schedule, min_p), get_self_correction(config))
        case _:
            return Gidd_independent_steps(model, noise_schedule, tokenizer, t_eps, get_sampling_strategy(config, tokenizer, noise_schedule, min_p), get_self_correction(config))

class SamplingStrategy(nn.Module):
    def __init__(self, model, noise_schedule, tokenizer, t_eps):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.t_eps = t_eps

    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        self.initial_z_t = initial_z_t.to(device, non_blocking=True)
        self.diffusion_mask = diffusion_mask.to(device, non_blocking=True)
        self.solution = solution.to(device, non_blocking=True)
        self.puzzle_conditioning = puzzle_conditioning
        self.num_denoising_steps = num_denoising_steps
        self.num_self_correction_steps = num_self_correction_steps
        self.max_length = max_length
        self.device = device

    @abstractmethod
    def stopping_criterion(self, i):
        raise NotImplementedError
    
    @abstractmethod
    def step(self, i, generation_info_handler):
        raise NotImplementedError


class Gidd_independent_steps(SamplingStrategy):
    def __init__(self, model, noise_schedule, tokenizer, t_eps, sampling_strategy, self_correction=None):
        super().__init__(model, noise_schedule, tokenizer, t_eps)
        self.sampling_strategy = sampling_strategy
        self.self_correction = self_correction

    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        self.ts = torch.linspace(0, 1, self.num_denoising_steps + 1, device=device).unsqueeze(-1)
        self.ts = (1 - 2 * self.t_eps) * self.ts + self.t_eps
        self.z_t = self.initial_z_t.clone()
        self.is_fully_unmasked = torch.zeros(self.initial_z_t.shape[0], dtype=torch.bool, device=device)
        self.has_self_corrected = False
    
    def stopping_criterion(self, i):
        if i < self.num_denoising_steps:
            return self.is_fully_unmasked.all() and self.self_correction is None
        else:
            return self.self_correction is None or self.has_self_corrected
        
    def step(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            self.is_fully_unmasked = (self.z_t[:, -self.max_length:] != self.tokenizer.mask_token_id).all(dim=1)
            if self.is_fully_unmasked.all():
                return
            t = self.ts[self.num_denoising_steps - 1 - i]
            s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
            logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)

            update_positions, next_z_t = self.sampling_strategy(probs, self.z_t, t, s, self.num_denoising_steps - 1 - i, self.diffusion_mask)

            update_positions = update_positions * self.diffusion_mask
            update_positions = update_positions & ~self.is_fully_unmasked.unsqueeze(-1)
            self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)
        elif self.self_correction is not None and not self.has_self_corrected:
            # TODO: change self-correction code to update one step at a time, such that history can be collected easily
            self.z_t, _ = self.self_correction(self.model, self.tokenizer, self.diffusion_mask, self.z_t, self.ts[0].item(), max_num_denoising_steps=self.num_self_correction_steps)
            self.has_self_corrected = True

class Gidd_flattened(SamplingStrategy):
    def __init__(self, model, noise_schedule, tokenizer, t_eps, sampling_strategy, self_correction=None):
        super().__init__(model, noise_schedule, tokenizer, t_eps)
        self.sampling_strategy = sampling_strategy
        self.self_correction = self_correction

    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        self.ts = torch.linspace(0, 1, self.num_denoising_steps + 1, device=device).unsqueeze(-1)
        self.ts = (1 - 2 * self.t_eps) * self.ts + self.t_eps
        self.z_t = self.initial_z_t.clone()
        self.max_score = torch.zeros_like(self.z_t, dtype=torch.float, device=device)
        self.is_fully_unmasked = torch.zeros(self.initial_z_t.shape[0], dtype=torch.bool, device=device)
        self.has_self_corrected = False
    
    def stopping_criterion(self, i):
        if i < self.num_denoising_steps:
            return self.is_fully_unmasked.all() and self.self_correction is None
        else:
            return self.self_correction is None or self.has_self_corrected
        
    def step(self, i, generation_info_handler):
        if i < self.num_denoising_steps:
            self.is_fully_unmasked = (self.z_t[:, -self.max_length:] != self.tokenizer.mask_token_id).all(dim=1)
            if self.is_fully_unmasked.all():
                return
            t = self.ts[self.num_denoising_steps - 1 - i]
            s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
            logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
            logits[..., self.tokenizer.mask_token_id:] = -1e6
            probs = logits.softmax(-1)

            update_positions, next_z_t, self.max_score = self.sampling_strategy(probs, self.z_t, t, s, self.num_denoising_steps - 1 - i, self.num_denoising_steps, self.diffusion_mask, self.max_score)

            update_positions = update_positions * self.diffusion_mask
            update_positions = update_positions & ~self.is_fully_unmasked.unsqueeze(-1)
            self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)
        elif self.self_correction is not None and not self.has_self_corrected:
            # TODO: change self-correction code to update one step at a time, such that history can be collected easily
            self.z_t, _ = self.self_correction(self.model, self.tokenizer, self.diffusion_mask, self.z_t, self.ts[0].item(), max_num_denoising_steps=self.num_self_correction_steps)
            self.has_self_corrected = True

class Gidd_prob_to_recover_data(SamplingStrategy):
    def __init__(self, config, model, noise_schedule, tokenizer, t_eps):
        super().__init__(model, noise_schedule, tokenizer, t_eps)
        self.config = config

    def initialize(self, initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device):
        super().initialize(initial_z_t, diffusion_mask, solution, puzzle_conditioning, num_denoising_steps + num_self_correction_steps, 0, max_length, device)
        # super().initialize(initial_z_t, diffusion_mask, puzzle_conditioning, num_denoising_steps, num_self_correction_steps, max_length, device)
        self.ts = torch.linspace(0, 1, self.num_denoising_steps + 1, device=device).unsqueeze(-1)
        self.ts = (1 - 2 * self.t_eps) * self.ts + self.t_eps
        self.z_t = self.initial_z_t.clone()
        self.p_zs_x = torch.zeros_like(self.initial_z_t, dtype=torch.float, device=device)
        self.is_fully_denoised = torch.zeros(self.initial_z_t.shape[0], dtype=torch.bool, device=device)
        self.not_mask_token_id_tensor = torch.zeros((1, 1), dtype=initial_z_t.dtype, device=device)
        self.mask_token_id_tensor = torch.ones((1, 1), dtype=initial_z_t.dtype, device=device) * self.tokenizer.mask_token_id
        # self.cum_p_not_denoised = 1.0

    def stopping_criterion(self, i):
        return self.is_fully_denoised.all() or i >= self.num_denoising_steps
    
    def step(self, i, generation_info_handler):
        t = self.ts[self.num_denoising_steps - 1 - i]
        s = self.ts[max(0, self.num_denoising_steps - 2 - i)]
        logits = self.model(self.z_t, t, puzzle_conditioning=self.puzzle_conditioning)
        logits[..., self.tokenizer.mask_token_id:] = -1e6
        probs = logits.softmax(-1)
        if i == self.num_denoising_steps + self.num_self_correction_steps - 1:
            update_positions = (self.z_t == self.tokenizer.mask_token_id)
            next_z_t = probs.argmax(-1)
        else:
            # denoising event: p(z_s = x, z_t != x)
            # p(z_s = x) = p(z_t = x) * p(z_s = x | z_t = x) + p(z_t != x) * p(z_s = x | z_t != x)

            # p(z_s = x, z_t != x) = q_t|s(z_t != x | z_s = x) * q_s(z_s = x | x) / q_t(z_t != x | x)
            # = (alpha_s + beta_pi_s_at_zt) * beta_pi_ts_at_zt / beta_pi_t_at_zt # independent of prediction for x

            # p(z_s = x | z_t = x) = q_t|s(z_t = x | z_s = x) * q_s(z_s = x | x) / q_t(z_t = x | x)
            # = (alpha_ts * z_s + beta_pi_ts) * (alpha_s + beta_pi_s) / (alpha_t + beta_pi_t)
            # use x_theta = probs for z_s and take the element at z_t:
            # (alpha_ts * x_theta_at_zt + beta_pi_ts_at_zt) * (alpha_s + beta_pi_s_at_zt) / (alpha_t + beta_pi_t_at_zt)

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
                weight = 0.5
                p_zt_x = weight * p_zt_x_model + (1 - weight) * self.p_zs_x

            p_zs_x_and_zt_x = p_zt_x * (alpha_s + beta_pi_s_at_not_m) * (alpha_ts + beta_pi_ts_at_not_m) / (alpha_t + beta_pi_t_at_not_m)
            p_zs_x_and_zt_nx = (1 - p_zt_x) * (alpha_s + beta_pi_s_at_not_m) * beta_pi_ts_at_zt / beta_pi_t_at_zt
            
            # use p_zs_x as stopping criterion
            denoised = p_zt_x > 0.9
            denoised = denoised | ~self.diffusion_mask.bool()
            self.is_fully_denoised = denoised.all(dim=1)

            # z_t based on forward distribution # issue: p_zs_x_and_zt_nx is small, not enough time for all positions to unmask (even with more steps, tried 300)
            # p_zs_x_and_zt_x = (alpha_s + beta_pi_s_at_not_m) * (alpha_ts + beta_pi_ts_at_not_m)
            # p_zs_x_and_zt_nx = (alpha_s + beta_pi_s_at_not_m) * (beta_pi_ts_at_m + (vocab_size_semantically - 2) * beta_pi_ts_at_not_m).expand_as(self.z_t)
            # p_zs_x = p_zs_x_and_zt_x + p_zs_x_and_zt_nx

            # decide whether to update based on p(z_s = x && z_t != x)
            if self.config.position_sampling == "independent":
                if self.config.position_metric == "p_denoise":
                    update_positions = torch.rand_like(p_zs_x_and_zt_nx) < p_zs_x_and_zt_nx
                elif self.config.position_metric == "confident_and_p_denoise":
                    confident_and_p_denoise = probs.max(-1).values * p_zs_x_and_zt_nx
                    update_positions = torch.rand_like(confident_and_p_denoise) < confident_and_p_denoise
                elif self.config.position_metric == "confident_and_noisy": # might not make sense but for the sake of running the cross product of configuration options keep this
                    confident_and_noisy = probs.max(-1).values * (1 - p_zt_x)
                    update_positions = torch.rand_like(confident_and_noisy) < confident_and_noisy
                elif self.config.position_metric == "noisy":
                    noisy = 1 - p_zt_x
                    update_positions = torch.rand_like(noisy) < noisy
            elif self.config.position_sampling == "top_k":
                # updatable_positions = self.diffusion_mask
                updatable_positions = self.diffusion_mask & ~denoised
                if self.config.position_metric == "p_denoise":
                    update_positions = sample_position_top_k_gumbel(p_zs_x_and_zt_nx * updatable_positions, 1, 0) # select positions with top-k probabilities to have a denoising event
                elif self.config.position_metric == "confident_and_p_denoise":
                    update_positions = sample_position_top_k_gumbel(probs.max(-1).values * p_zs_x_and_zt_nx * updatable_positions, 1, 0)
                elif self.config.position_metric == "confident_and_noisy":
                    update_positions = sample_position_top_k_gumbel(probs.max(-1).values * (1 - p_zt_x) * updatable_positions, 1, 0)
                    # update_positions = sample_position_top_k_gumbel((probs.max(-1).values + (1 - self.p_zt_x)) * self.diffusion_mask, 1, 0)

            # update selected positions
            # token_sampling = "categorical"
            # token_sampling = "change_max"
            # token_sampling = "max"
            if self.config.token_sampling == "categorical":
                next_z_t = sample_categorical(probs, end_index=self.tokenizer.unk_token_id - 1) # sample categorically from predictions
            elif self.config.token_sampling == "change_max":
                next_z_t = sample_token_change_max(probs, self.z_t) # force change
            elif self.config.token_sampling == "max":
                next_z_t = sample_token_MDM_max(probs) # don't force change

            # TODO: decide whether to update based on p_zs_nx_and_zt_nx (transitions x' -> u and x' -> x', for x' != x (either x' = m or x' = u))
            
            update_positions = update_positions * self.diffusion_mask
            # update_positions = update_positions & ~self.is_fully_denoised.unsqueeze(-1) # only prevent updating if entire sample is currently believed to be denoised
            update_positions = update_positions & ~denoised # don't update positions that are currently believed to be denoised
            
            self.p_zs_x = p_zs_x_and_zt_x + p_zs_x_and_zt_nx
            if self.config.oracle == "recurrence" or self.config.oracle == "model_and_recurrence":
                self.p_zs_x = torch.where(update_positions.bool(), probs.gather(-1, next_z_t.unsqueeze(-1)).squeeze(-1), self.p_zs_x)
            elif self.config.oracle == "model_EMA":
                self.p_zs_x = torch.where(update_positions.bool(), probs.gather(-1, next_z_t.unsqueeze(-1)).squeeze(-1), p_zt_x)
            
        self.z_t = torch.where(update_positions.bool(), next_z_t, self.z_t)

        if generation_info_handler is not None:
            generation_info_handler.batch_step_history(self.z_t)
            if i != self.num_denoising_steps + self.num_self_correction_steps - 1:
                generation_info_handler.batch_step_marginals(self.z_t, p_zt_x, self.tokenizer.mask_token_id)
        # TODO: collect fraction of mask, noise-free and uniform tokens
        
        # sample_in_batch = 19
        # position_in_sample = 81 + 56
        # print(f"p_zt_x: {self.p_zt_x[sample_in_batch, position_in_sample]}")
        # self.cum_p_not_denoised *= (1 - p_zs_x_and_zt_nx[sample_in_batch, position_in_sample].item())
        # print(f"{i}, z_t: {self.z_t[sample_in_batch, position_in_sample].item()}")
        # print(f"{i}, num masks in z_t per sample: {(self.z_t[:, -81:] == self.tokenizer.mask_token_id).sum(-1)}")
        # print(f"{i}, max num masks in z_t per sample: {(self.z_t[:, -81:] == self.tokenizer.mask_token_id).sum(-1).max()}")
        # print(f"{i}, dice_roll: {dice_roll[sample_in_batch, position_in_sample].item()}")
        # if i != self.num_denoising_steps + self.num_self_correction_steps - 1:
            # print(f"{i}, x_theta_at_zt: {x_theta_at_zt[sample_in_batch, position_in_sample].item()}")
            # print(f"{i}, constant term x: {((alpha_s + beta_pi_s_at_not_m) * (alpha_ts + beta_pi_ts_at_not_m) / (alpha_t + beta_pi_t_at_not_m))[0, 0].item()}")
            # print(f"{i}, p_zs_x_and_zt_nx: {p_zs_x_and_zt_nx[sample_in_batch, position_in_sample].item()}")
        # print(f"{i}, cum_p_not_denoised: {self.cum_p_not_denoised}")