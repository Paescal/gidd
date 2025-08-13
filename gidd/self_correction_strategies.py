import pandas as pd
import hydra
import tqdm
import torch
from functools import partial

from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.utils import sample_categorical



class UpdatableMaskHandler:
    def __init__(self, initial_z_t):
        self.previous_z_t = None
        self.z_t = initial_z_t.clone()
        self.updatable_mask = torch.ones_like(self.z_t)
    
    def get_mask(self):
        return self.updatable_mask
    
    def update_mask(self, z_t_next):
        if self.previous_z_t is not None:
            if (self.previous_z_t == z_t_next).all():
                position_did_not_oscillate = self.z_t == z_t_next
                self.updatable_mask = self.updatable_mask * position_did_not_oscillate
            else:
                self.updatable_mask = torch.ones_like(self.z_t)
        self.previous_z_t = self.z_t
        self.z_t = z_t_next.clone()
# TODO?:Different version: If z_t_next is a state which has been seen before, remove the positions that changed from z_t to z_t_next from the updatable_mask for state z_t


class PositionBackoffHandler:
    def __init__(self, initial_z_t, backoff_factor=0.5, recovery_factor_fast=0.5, recovery_factor_slow=1.0, recovery_type='slow'):
        self.z_t = initial_z_t.clone()
        self.backoff_factor = backoff_factor
        self.recovery_factor_fast = recovery_factor_fast
        self.recovery_factor_slow = recovery_factor_slow
        self.recovery_type = recovery_type
        self.position_weights = torch.ones_like(self.z_t, dtype=torch.float32)
    
    def get_weights(self):
        return self.position_weights
    
    def update_weights(self, z_t_next):
        updated_positions = (self.z_t != z_t_next)
        degraded_weights = self.position_weights * (1 - self.backoff_factor)
        if self.recovery_type == 'fast':
            recovered_weights = 1 - (1 - self.position_weights) * self.recovery_factor_fast
        elif self.recovery_type == 'slow':
            recovered_weights = self.position_weights * (1 + self.recovery_factor_slow)
            recovered_weights = torch.clamp(recovered_weights, min=0, max=1)
        else:
            raise ValueError(f"Unknown recovery type: {self.recovery_type}")
        self.position_weights = torch.where(updated_positions, degraded_weights, recovered_weights)
        self.z_t = z_t_next.clone()
    
    

# def correction_step_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step):
#     logits = model(z_t, t)
#     logits[..., tokenizer.mask_token_id] = -1e6

#     p_t = (logits / temp).softmax(-1)

#     z_tm1 = sample_categorical(p_t)
#     score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1) * diffusion_mask

#     ids = torch.topk(score, tokens_per_step, dim=-1).indices
#     z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

#     acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
#     return torch.where(diffusion_mask.to(dtype=bool), z_tm1, z_t), acc

# def self_correction_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
#     t = torch.full((z_t.shape[0],), device=z_t.device, fill_value=t)

#     logits = model(z_t, t)
#     logits[..., tokenizer.mask_token_id] = -1e6
#     init_acc = ((z_t == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
#     print(f"Initial accuracy: {init_acc}")
#     if (init_acc == 1).all():
#         return z_t, []

#     history = [] if keep_history else None
    
#     max_acc = init_acc.clone()
#     curr_patience = torch.zeros_like(max_acc)
#     # converged = 0
#     # early_stopped = 0
#     for i in range(max_num_denoising_steps):
#         z_t_next, acc = correction_step_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step)
#         # TODO: accuracy is a tensor of shape (batch_size, ), so fix code accordingly

#         max_acc = torch.where(acc > max_acc, acc, max_acc)
#         curr_patience = torch.where(acc <= max_acc, curr_patience + 1, torch.zeros_like(curr_patience))
#         if (curr_patience > max_patience).all():
#             # early_stopped = 1
#             break
#         if (z_t == z_t_next).all():
#             # converged = 1
#             break
#         z_t = z_t_next
#         if keep_history:
#             history.append(z_t_next.clone())

#     # num_changes = (initial_z_t != z_t).sum().item()
#     print(f"Final accuracy: {acc}")
#     return z_t, history


def get_self_correction(config):
    match config.sampling.self_correction:
        case "none":
            return None
        case "original":
            temp = 1
            tokens_per_step = 1
            return partial(self_correction_original, temp=temp, tokens_per_step=tokens_per_step)
        case "oscillation_prevention_fast":
            temp = 1
            tokens_per_step = 1
            backoff_factor=0.5
            recovery_factor_fast=0.2
            recovery_type='fast'
            return partial(self_correction_original_oscillation_prevention, temp=temp, tokens_per_step=tokens_per_step, backoff_factor=backoff_factor, recovery_factor_fast=recovery_factor_fast, recovery_type=recovery_type)
        case "oscillation_prevention_slow":
            temp = 1
            tokens_per_step = 1
            backoff_factor=0.5
            recovery_factor_slow=0.4142
            recovery_type='slow'
            return partial(self_correction_original_oscillation_prevention, temp=temp, tokens_per_step=tokens_per_step, backoff_factor=backoff_factor, recovery_factor_slow=recovery_factor_slow, recovery_type=recovery_type)
        case "keep_where_confident":
            temp = 1
            tokens_per_step = 1
            return partial(self_correction_keep_where_confident, temp=temp, tokens_per_step=tokens_per_step)
    pass


def correction_step_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id:] = -1e6

    p_t = (logits / temp).softmax(-1)

    z_tm1 = sample_categorical(p_t)
    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1) * diffusion_mask

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    return torch.where(diffusion_mask.to(dtype=bool), z_tm1, z_t), acc

def self_correction_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
    samples = []
    history = []
    t = torch.full((1,), device=z_t.device, fill_value=t)
    max_denoising_steps_in_batch = 0
    for sample, sample_diffusion_mask in zip(z_t, diffusion_mask, strict=True):
        sample = sample.unsqueeze(0)
        sample_diffusion_mask = sample_diffusion_mask.unsqueeze(0)
        sample_history = []
        logits = model(sample, t)
        logits[..., tokenizer.mask_token_id:] = -1e6
        
        init_acc = ((sample == logits.argmax(-1)) * sample_diffusion_mask).sum(-1) / sample_diffusion_mask.sum(-1)
        if init_acc == 1:
            samples.append(sample)
            if keep_history:
                history.append([])
            continue
        max_acc = 0
        curr_patience = 0
        # converged = 0
        # early_stopped = 0
        for i in range(max_num_denoising_steps):
            sample_next, acc = correction_step_original(model, tokenizer, sample_diffusion_mask, sample, t, temp, tokens_per_step)

            if acc > max_acc:
                max_acc = acc
                curr_patience = 0
            else:
                curr_patience += 1
                if curr_patience > max_patience:
                    # early_stopped = 1
                    # print(f"Early stopped at step {i + 1}")
                    break

            if (sample == sample_next).all():
                # converged = 1
                # print(f"Converged at step {i + 1}")
                break
            if i + 1 > max_denoising_steps_in_batch:
                max_denoising_steps_in_batch = i + 1
            sample = sample_next
            if keep_history:
                sample_history.append(sample_next.clone())
        samples.append(sample)
        if keep_history:
            history.append(sample_history)

    z_t = torch.cat(samples, dim=0)
    if keep_history:
        if max_denoising_steps_in_batch == 0:
            history = []
        else:
            for i in range(len(history)):
                history[i] = history[i] + [z_t[i].unsqueeze(0)] * (max_denoising_steps_in_batch - len(history[i]))
                history[i] = torch.cat(history[i], dim=0)
            history = torch.stack(history, dim=1)
            history = list(torch.unbind(history, dim=0))
    return z_t, history


def correction_step_original_oscillation_prevention(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, position_backoff_handler):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id:] = -1e6

    p_t = (logits / temp).softmax(-1)

    z_tm1 = sample_categorical(p_t)
    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1) * diffusion_mask * position_backoff_handler.get_weights()

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    z_tm1 = torch.where(diffusion_mask.to(dtype=bool), z_tm1, z_t)
    position_backoff_handler.update_weights(z_tm1)
    return z_tm1, acc

def self_correction_original_oscillation_prevention(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, backoff_factor=0.5, recovery_factor_fast=0.5, recovery_factor_slow=1.0, recovery_type='slow', keep_history=False):
    samples = []
    history = []
    t = torch.full((1,), device=z_t.device, fill_value=t)
    max_denoising_steps_in_batch = 0
    for sample, sample_diffusion_mask in zip(z_t, diffusion_mask, strict=True):
        sample = sample.unsqueeze(0)
        sample_diffusion_mask = sample_diffusion_mask.unsqueeze(0)
        sample_history = []
        logits = model(sample, t)
        logits[..., tokenizer.mask_token_id:] = -1e6
        
        init_acc = ((sample == logits.argmax(-1)) * sample_diffusion_mask).sum(-1) / sample_diffusion_mask.sum(-1)
        if init_acc == 1:
            samples.append(sample)
            if keep_history:
                history.append([])
            continue
        max_acc = 0
        curr_patience = 0
        # converged = 0
        # early_stopped = 0
        position_backoff_handler = PositionBackoffHandler(sample, backoff_factor=backoff_factor, recovery_factor_fast=recovery_factor_fast, recovery_factor_slow=recovery_factor_slow, recovery_type=recovery_type)
        for i in range(max_num_denoising_steps):
            sample_next, acc = correction_step_original_oscillation_prevention(model, tokenizer, sample_diffusion_mask, sample, t, temp, tokens_per_step, position_backoff_handler)

            if acc > max_acc:
                max_acc = acc
                curr_patience = 0
            else:
                curr_patience += 1
                if curr_patience > max_patience:
                    # early_stopped = 1
                    # print(f"Early stopped at step {i + 1}")
                    break

            if (sample == sample_next).all():
                # converged = 1
                # print(f"Converged at step {i + 1}")
                break
            if i + 1 > max_denoising_steps_in_batch:
                max_denoising_steps_in_batch = i + 1
            sample = sample_next
            if keep_history:
                sample_history.append(sample_next.clone())
        samples.append(sample)
        if keep_history:
            history.append(sample_history)

    z_t = torch.cat(samples, dim=0)
    if keep_history:
        if max_denoising_steps_in_batch == 0:
            history = []
        else:
            for i in range(len(history)):
                history[i] = history[i] + [z_t[i].unsqueeze(0)] * (max_denoising_steps_in_batch - len(history[i]))
                history[i] = torch.cat(history[i], dim=0)
            history = torch.stack(history, dim=1)
            history = list(torch.unbind(history, dim=0))
    return z_t, history


def correction_step_keep_where_confident(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id:] = -1e6
    probs = logits.softmax(-1)
    values, indices = torch.max(probs, dim=-1)
    scores = torch.where(z_t == indices, 0, values) * diffusion_mask
    update_positions = torch.topk(scores, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.clone()
    z_tm1 = z_tm1.scatter(-1, update_positions, indices.gather(-1, update_positions))
    acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    return torch.where(diffusion_mask.to(dtype=bool), z_tm1, z_t), acc

def self_correction_keep_where_confident(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
    samples = []
    history = []
    t = torch.full((1,), device=z_t.device, fill_value=t)
    max_denoising_steps_in_batch = 0
    for sample, sample_diffusion_mask in zip(z_t, diffusion_mask, strict=True):
        sample = sample.unsqueeze(0)
        sample_diffusion_mask = sample_diffusion_mask.unsqueeze(0)
        sample_history = []
        logits = model(sample, t)
        logits[..., tokenizer.mask_token_id:] = -1e6
        
        init_acc = ((sample == logits.argmax(-1)) * sample_diffusion_mask).sum(-1) / sample_diffusion_mask.sum(-1)
        if init_acc == 1:
            samples.append(sample)
            if keep_history:
                history.append([])
            continue
        max_acc = 0
        curr_patience = 0
        # converged = 0
        # early_stopped = 0
        for i in range(max_num_denoising_steps):
            sample_next, acc = correction_step_keep_where_confident(model, tokenizer, sample_diffusion_mask, sample, t, temp, tokens_per_step)

            if acc > max_acc:
                max_acc = acc
                curr_patience = 0
            else:
                curr_patience += 1
                if curr_patience > max_patience:
                    # early_stopped = 1
                    # print(f"Early stopped at step {i + 1}")
                    break

            if (sample == sample_next).all():
                # converged = 1
                # print(f"Converged at step {i + 1}")
                break
            if i + 1 > max_denoising_steps_in_batch:
                max_denoising_steps_in_batch = i + 1
            sample = sample_next
            if keep_history:
                sample_history.append(sample_next.clone())
        samples.append(sample)
        if keep_history:
            history.append(sample_history)

    z_t = torch.cat(samples, dim=0)
    if keep_history:
        if max_denoising_steps_in_batch == 0:
            history = []
        else:
            for i in range(len(history)):
                history[i] = history[i] + [z_t[i].unsqueeze(0)] * (max_denoising_steps_in_batch - len(history[i]))
                history[i] = torch.cat(history[i], dim=0)
            history = torch.stack(history, dim=1)
            history = list(torch.unbind(history, dim=0))
    return z_t, history






# def correction_step_max(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, updatable_mask):
#     logits = model(z_t, t)
#     logits[..., tokenizer.mask_token_id:] = -1e6
#     logits.scatter_(-1, z_t.unsqueeze(-1), -1e6)

#     score, z_tm1 = torch.max(logits, dim=-1)

#     # TODO: Use a threshold or something to allow the step to not change anything if the model is already confident
#     # Or maybe not, because the step is only called if the accuracy is not 1

#     score = score.softmax(-1) * updatable_mask * diffusion_mask
#     # print(f"score: {score[0, -81:]}")
#     # print(f"updatable_mask: {updatable_mask[0, -81:]}")

#     # only if not all scores zero
#     top_k_threshold = torch.topk(score, tokens_per_step, dim=-1).values[-1]
#     update_positions = score >= top_k_threshold
#     update_positions = update_positions * updatable_mask * diffusion_mask
#     z_tm1 = torch.where(update_positions.to(dtype=bool), z_tm1, z_t)

#     acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
#     print(f"z_t: {z_t[0, -81:]}")
#     # print(f"update_positions: {update_positions[0, -81:]}")
#     # print(f"z_tm1: {z_tm1[0, -81:]}")
#     # print(f"logits.argmax(-1): {logits.argmax(-1)[0, -81:]}")
#     print(f"logits: {logits[0, -1, :11]}")
#     print(f"acc: {acc[0]}")
#     return z_tm1, acc

# def self_correction_max(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
#     t = torch.full((z_t.shape[0],), device=z_t.device, fill_value=t)

#     logits = model(z_t, t)
#     logits[..., tokenizer.mask_token_id:] = -1e6
#     init_acc = ((z_t == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
#     self_correction_not_done = (init_acc < 1)
#     print(f"Initial accuracy: {init_acc[0]}")
#     print(f"Initial logits: {logits[0, -1, :11]}")
#     print(f"Initial z_t: {z_t[0, -81:]}")
#     if (init_acc == 1).all():
#         return z_t, []
    
#     history = [] if keep_history else None
    
#     max_acc = init_acc.clone()
#     curr_patience = torch.zeros_like(max_acc)
#     # converged = 0
#     # early_stopped = 0
#     updatable_mask_handler = UpdatableMaskHandler(z_t)
#     for i in range(max_num_denoising_steps):
#         # print(f"Diffusion mask: {diffusion_mask[0, -81:]}")
#         # print(f"current accuracy: {torch.where(diffusion_mask.to(dtype=bool), (z_t == logits.argmax(-1)), 1)[0, -81:]}")
#         updatable_mask = updatable_mask_handler.get_mask()
#         # print(f"updatable_mask from handler: {updatable_mask[0, -81:]}")
#         updatable_mask = updatable_mask * (self_correction_not_done).unsqueeze(-1)
#         # print(f"updatable_mask after applying self_correction_not_done: {updatable_mask[0, -81:]}")
#         # print(f"updatable_mask: {updatable_mask[0, -81:]}, self_correction_not_done: {self_correction_not_done[0]}")
#         # print(f"z_t: {z_t[0, -81:]}")
#         z_t_next, acc = correction_step_max(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, updatable_mask)
#         # print(f"z_t_next: {z_t_next[0, -81:]}")
#         # print(f"new accuracy: {acc[0]}")

#         self_correction_not_done = (acc < 1)
#         max_acc = torch.where(acc > max_acc, acc, max_acc)
#         curr_patience = torch.where(acc > max_acc, torch.zeros_like(curr_patience), curr_patience + 1)
#         if (curr_patience > max_patience).all():
#             # early_stopped = 1
#             break
#         if (z_t == z_t_next).all():
#             # converged = 1
#             break
#         updatable_mask_handler.update_mask(z_t_next)
#         z_t = z_t_next
        
#         if keep_history:
#             history.append(z_t_next.clone())

#     # num_changes = (initial_z_t != z_t).sum().item()
#     print(f"Final accuracy: {acc}")
#     return z_t, history

# TODO: Try margin

