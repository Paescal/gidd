import pandas as pd
import hydra
import tqdm
import torch

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


def correction_step_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)

    z_tm1 = sample_categorical(p_t)
    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1) * diffusion_mask

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    return torch.where(diffusion_mask, z_tm1, z_t), acc

def correction_step_max(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, updatable_mask):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6
    logits.scatter_(-1, z_t.unsqueeze(-1), -1e6)

    score, z_tm1 = torch.max(logits, dim=-1)

    # TODO: Use a threshold or something to allow the step to not change anything if the model is already confident
    # Or maybe not, because the step is only called if the accuracy is not 1

    score = score.softmax(-1) * updatable_mask * diffusion_mask

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = ((z_tm1 == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    return z_tm1, acc


def self_correction_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
    max_acc = 0
    curr_patience = 0
    initial_z_t = z_t.clone()
    t = torch.full((z_t.shape[0],), device=z_t.device, fill_value=t)

    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6
    # self_correction_done = torch.where(diffusion_mask, z_t == logits.argmax(-1), True).all(dim=-1)
    init_acc = ((z_t == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    print(f"Initial accuracy: {init_acc}")
    if (init_acc == 1).all():
        return z_t, []

    history = [] if keep_history else None
    
    converged = 0
    early_stopped = 0
    for i in range(max_num_denoising_steps):
        z_t_next, acc = correction_step_original(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step)
        # TODO: accuracy is a tensor of shape (batch_size, ), so fix code accordingly

        if acc > max_acc:
            max_acc = acc
            curr_patience = 0
        else:
            curr_patience += 1
            if curr_patience > max_patience:
                early_stopped = 1
                break

        if (z_t == z_t_next).all():
            converged = 1
            break
        z_t = z_t_next
        if keep_history:
            history.append(z_t_next.clone())

    num_changes = (initial_z_t != z_t).sum().item()
    return z_t, history

def self_correction_max(model, tokenizer, diffusion_mask, z_t, t, temp, tokens_per_step, max_num_denoising_steps=20, max_patience=10, keep_history=False):
    max_acc = 0
    curr_patience = 0
    initial_z_t = z_t.clone()
    t = torch.full((z_t.shape[0],), device=z_t.device, fill_value=t)

    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6
    init_acc = ((z_t == logits.argmax(-1)) * diffusion_mask).sum(-1) / diffusion_mask.sum(-1)
    print(f"Initial accuracy: {init_acc}")
    if (init_acc == 1).all():
        return z_t, []
    
    history = [] if keep_history else None
    
    converged = 0
    early_stopped = 0
    updatable_mask_handler = UpdatableMaskHandler(initial_z_t)
    for i in range(max_num_denoising_steps):
        z_t_next, acc = correction_step_max(model, tokenizer, z_t, t, temp, tokens_per_step, updatable_mask_handler.get_mask())

        if acc > max_acc:
            max_acc = acc
            curr_patience = 0
        else:
            curr_patience += 1
            if curr_patience > max_patience:
                early_stopped = 1
                break

        if (z_t == z_t_next).all():
            converged = 1
            break
        if acc == 1:
            converged = 1
            z_t = z_t_next
            break
        updatable_mask_handler.update_mask(z_t_next)
        z_t = z_t_next
        
        if keep_history:
            history.append(z_t_next.clone())

    num_changes = (initial_z_t != z_t).sum().item()
    return z_t, history