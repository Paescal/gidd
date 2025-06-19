import torch
from functools import partial

from gidd.utils import position_metric_MDM_max, position_metric_MDM_margin, all_positions, sample_positions_independently, sample_position_top_k_gumbel, sample_token_MDM_max, sample_token_MDM_categorical, sample_categorical


def get_score_position(config, tokenizer):
    match config.sampling.score_position:
        case "MDM_max":
            return partial(position_metric_MDM_max, tokenizer=tokenizer)
        case "MDM_margin":
            return partial(position_metric_MDM_margin, tokenizer=tokenizer)

def get_select_position(config, tokenizer):
    match config.sampling.select_position:
        case "all":
            return all_positions
        case "independent":
            return sample_positions_independently
        case "top_k_gumbel":
            return partial(sample_position_top_k_gumbel, tokenizer=tokenizer)

def get_update_token(config, tokenizer):
    match config.sampling.update_token:
        case "MDM_max":
            return partial(sample_token_MDM_max, tokenizer=tokenizer)
        case "MDM_categorical":
            return partial(sample_token_MDM_categorical, tokenizer=tokenizer)
        case "categorical":
            return sample_categorical

def get_sampling_strategy(config, tokenizer):
    return partial(mdlm_vanilla(update_token=get_update_token(config, tokenizer)))
    match config.sampling.strategy:
        case "mdlm_vanilla":
            return partial(mdlm_vanilla(update_token=get_update_token(config, tokenizer)))
        case "mdlm_adaptive_score_select_update":
            return partial(mdlm_adaptive_score_select_update(
                score_position=get_score_position(config, tokenizer),
                select_position=get_select_position(config, tokenizer),
                update_token=get_update_token(config, tokenizer)))



@torch.no_grad()
def mdlm_vanilla(update_token, probs, z_t, t, tm1, diffusion_mask, eps=1e-4):
    def get_sigmas(self, t, eps=1e-4):
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
    update_positions_from_strategy = sample_positions_independently(z_t, prob_unmask_this_step)
    z_tm1 = update_token(probs)
    return update_positions_from_strategy, z_tm1

@torch.no_grad()
def mdlm_adaptive_score_select_update(score_position, select_position, update_token, probs, z_t, t, tm1, diffusion_mask, eps=1e-4):
    score = score_position(z_t, probs) * diffusion_mask
    update_positions = select_position(score)
    z_tm1 = update_token(probs)
    return update_positions, z_tm1

@torch.no_grad()
def gidd_vanilla_original(probs, z_t, t, s, diffusion_mask):
    pass

@torch.no_grad()
def gidd_vanilla_split(probs, z_t, t, s, diffusion_mask):
    pass

@torch.no_grad()
def gidd_adaptive_score_select_update(probs, z_t, t, s, diffusion_mask):
    pass

@torch.no_grad()
def gidd_adaptive_change_vs_unmask(probs, z_t, t, s, diffusion_mask):
    pass
