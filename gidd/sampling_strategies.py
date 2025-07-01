import torch
import torch.nn.functional as F
from functools import partial

from gidd.utils import (
    position_score_change_max,
    position_score_change_margin,
    position_metric_MDM_max,
    position_metric_MDM_margin,
    all_positions,
    sample_positions_independently,
    sample_position_top_k_gumbel,
    sample_token_change_max,
    sample_token_change_categorical,
    sample_token_MDM_max,
    sample_token_MDM_categorical,
    sample_categorical,
)


def get_score_position(config, tokenizer):
    match config.sampling.score_position:
        case "MDM_max":
            return partial(position_metric_MDM_max, tokenizer=tokenizer)
        case "MDM_margin":
            return partial(position_metric_MDM_margin, tokenizer=tokenizer)
        case "change_max":
            return partial(position_score_change_max, tokenizer=tokenizer)
        case "change_margin":
            return partial(position_score_change_margin, tokenizer=tokenizer)

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
            return all_positions
        case "independent":
            return sample_positions_independently
        case "top_k_gumbel":
            return partial(sample_position_top_k_gumbel, k=config.sampling.k, gumbel_noise_coefficient=config.sampling.gumbel_noise_coefficient)

def get_update_token(config, tokenizer, arg_name='update_token'):
    match arg_name:
        case 'update_token':
            arg = config.sampling.update_token
        case 'update_token_change':
            arg = config.sampling.update_token_change
        case 'update_token_unmask':
            arg = config.sampling.update_token_unmask

    match arg:
        case "MDM_max":
            return partial(sample_token_MDM_max, tokenizer=tokenizer)
        case "MDM_categorical":
            return partial(sample_token_MDM_categorical, tokenizer=tokenizer)
        case "change_max":
            return partial(sample_token_change_max, tokenizer=tokenizer)
        case "change_categorical":
            return partial(sample_token_change_categorical, tokenizer=tokenizer)
        case "categorical":
            return sample_categorical

def get_sampling_strategy(config, tokenizer, noise_schedule=None, min_p=None):
    match config.sampling.strategy:
        case "mdlm_vanilla":
            return partial(mdlm_vanilla,
                           update_token=get_update_token(config, tokenizer),)
        case "mdlm_adaptive_score_select_update":
            return partial(mdlm_adaptive_score_select_update,
                           score_position=get_score_position(config, tokenizer),
                           select_position=get_select_position(config),
                           update_token=get_update_token(config, tokenizer),)
        case "gidd_vanilla_original":
            return partial(gidd_vanilla_original,
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,)
        case "gidd_vanilla_split":
            return partial(gidd_vanilla_split,
                           noise_schedule=noise_schedule,
                           update_token=get_update_token(config, tokenizer),)
        case "gidd_vanilla_independent_change":
            return partial(gidd_vanilla_independent_change,
                           update_token=get_update_token(config, tokenizer),
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,)
        case "gidd_adaptive_score_select_update":
            return partial(gidd_adaptive_score_select_update,
                           score_position=get_score_position(config, tokenizer),
                           select_position=get_select_position(config),
                           update_token=get_update_token(config, tokenizer),)
        case "gidd_adaptive_change_vs_unmask":
            return partial(gidd_adaptive_change_vs_unmask,
                           tokenizer=tokenizer,
                           noise_schedule=noise_schedule,
                           min_p=min_p,
                           score_position=get_score_position(config, tokenizer),
                           select_position_change=get_select_position(config, arg_name='select_position_change'),
                           select_position_unmask=get_select_position(config, arg_name='select_position_unmask'),
                           update_token_change=get_update_token(config, tokenizer, arg_name='update_token_change'),
                           update_token_unmask=get_update_token(config, tokenizer, arg_name='update_token_unmask'),)
        case "gidd_change_based_on_model_confidence_to_change":
            return partial(gidd_change_based_on_model_confidence_to_change,
                           score_position=get_score_position(config, tokenizer),
                           select_position=get_select_position(config),
                           update_token=get_update_token(config, tokenizer),)


@torch.no_grad()
def mdlm_vanilla(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, update_token=None):
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
    z_tm1 = update_token(probs)
    return update_positions, z_tm1

@torch.no_grad()
def mdlm_adaptive_score_select_update(probs, z_t, t, tm1, diffusion_mask, eps=1e-4, score_position=None, select_position=None, update_token=None):
    score = score_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    z_tm1 = update_token(probs)
    return update_positions, z_tm1

@torch.no_grad()
def gidd_vanilla_original(probs, z_t, t, s, diffusion_mask, tokenizer, noise_schedule, min_p):
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
    
    update_positions = torch.ones_like(z_t, dtype=torch.bool)
    next_z_t = sample_categorical(q_st, end_index=tokenizer.unk_token_id - 1)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_vanilla_split(probs, z_t, t, s, diffusion_mask, noise_schedule, update_token):
    alpha_t, _ = noise_schedule.get_alpha_betapi(t)
    alpha_s, _ = noise_schedule.get_alpha_betapi(s)
    
    update_positions = sample_positions_independently(z_t, (alpha_s - alpha_t) / (1 - alpha_t))
    print(f"prob to update: {(alpha_s - alpha_t) / (1 - alpha_t)}")
    next_z_t = update_token(probs)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_vanilla_independent_change(probs, z_t, t, s, diffusion_mask, tokenizer, noise_schedule, min_p, update_token):
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
    next_z_t = update_token(probs, z_t)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_adaptive_score_select_update(probs, z_t, t, s, diffusion_mask, score_position, select_position, update_token):
    score = score_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    next_z_t = update_token(probs)
    return update_positions, next_z_t

@torch.no_grad()
def gidd_adaptive_change_vs_unmask(probs, z_t, t, s, diffusion_mask, tokenizer, noise_schedule, min_p, score_position, select_position_change, select_position_unmask, update_token_change, update_token_unmask):
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
    # print(f"Shape of probs to not unmask: {probs_to_not_unmask.shape}, beta_pi_s_at_mask: {beta_pi_s_at_mask.shape}")
    # print(f"probs to not unmask: {1 - probs_to_unmask[0]} ( = {beta_pi_s_at_mask.item()} / {beta_pi_t_at_mask.item()})")
    # print(f"probs to not change: {q_st_at_zt[0, token_start:token_end]}")
    # print(f"q_ts factor: {(alpha_t / alpha_s + beta_pi_t_at_mask - alpha_t / alpha_s * beta_pi_s_at_mask)[0, 0].item()}")
    probs_to_unmask = probs_to_unmask.unsqueeze(-1).expand_as(probs_to_change)

    probs_to_change = probs_to_change * diffusion_mask
    probs_to_unmask = probs_to_unmask * diffusion_mask

    positions_to_change = (probs_to_change > probs_to_unmask + 1e-6)
    # print(f"z_t: {z_t[0, token_start:token_end]}")
    # print(f"probs_to_change: {probs_to_change[0, token_start:token_end]}")
    # print(f"probs_to_unmask: {probs_to_unmask[0, token_start:token_end]}")

    if positions_to_change.any():
        # print("Changing tokens")
        # print(f"Positions to change: {positions_to_change[0, token_start:token_end]}")
        score = probs_to_change * positions_to_change.to(dtype=probs_to_change.dtype)
        score = score * diffusion_mask
        update_positions = select_position_change(score)
        next_z_t = update_token_change(probs, z_t)
    else:
        # print("Unmasking tokens")
        score = score_position(z_t, probs)
        score = score * diffusion_mask
        update_positions = select_position_unmask(score)
        next_z_t = update_token_unmask(probs)
    return update_positions, next_z_t

def gidd_change_based_on_model_confidence_to_change(probs, z_t, t, s, diffusion_mask, score_position, select_position, update_token):
    score = score_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    next_z_t = update_token(probs, z_t)
    return update_positions, next_z_t
