# from: https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py

import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
  import flash_attn
  import flash_attn.layers.rotary
  has_flash_attn = True
except ImportError:
  torch.backends.cuda.enable_flash_sdp(enabled=True)
  has_flash_attn = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


# @torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


# @torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


# @torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000, max_seq_len=512):
    super().__init__()
    self.dim = dim
    self.base = base
    self.max_seq_len = max_seq_len
    self.precompute()

  def precompute(self):
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
    t = torch.arange(self.max_seq_len).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq.clone())
    emb = torch.cat((freqs, freqs), dim=-1)
    # dims are: batch, seq_len, qkv, head, dim
    cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
    sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
    # This makes the transformation on v an identity.
    cos_cached[:,:,2,:,:].fill_(1.)
    sin_cached[:,:,2,:,:].fill_(0.)

    self.register_buffer('cos_cached', cos_cached)
    self.register_buffer('sin_cached', sin_cached)

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  if has_flash_attn:
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
  else:
    print(f"Applying rotary pos emb with shapes: qkv: {qkv.shape}, cos: {cos.shape}")
    return (qkv * cos) + (rotate_half(qkv) * sin)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale) + shift


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def flops(self, seq_len=128):
    per_token_flops = 0
    per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
    per_token_flops += 2 * seq_len * self.dim  # softmax attention
    per_token_flops += 2 * self.dim * self.dim  # attn_out
    per_token_flops += 2 * 2 * self.dim * 4 * self.dim  # mlp
    flops = per_token_flops * seq_len
    flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
    return flops

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    cos, sin = rotary_cos_sin
    qkv = apply_rotary_pos_emb(
      qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    
    if has_flash_attn:
      qkv = rearrange(qkv, 'b s ... -> (b s) ...')
      if seqlens is None:
        cu_seqlens = torch.arange(
          0, (batch_size + 1) * seq_len, step=seq_len,
          dtype=torch.int32, device=qkv.device)
      else:
        cu_seqlens = seqlens.cumsum(-1)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=False)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
      x = F.scaled_dot_product_attention(q, k, v)
      
      x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x

class DDiTBlockCross(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def flops(self, seq_len=128):
    per_token_flops = 0
    per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
    per_token_flops += 2 * seq_len * self.dim  # softmax attention
    per_token_flops += 2 * self.dim * self.dim  # attn_out
    per_token_flops += 2 * 2 * self.dim * 4 * self.dim  # mlp
    flops = per_token_flops * seq_len
    flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
    return flops

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, context, rotary_cos_sin, c, seqlens=None):
    """
    x: (batch, seq, dim) - the input to attend (queries)
    context: (batch, context_seq, dim) - the memory to attend to (keys/values)
    rotary_cos_sin: (cos, sin) for queries (x)
    c: conditioning
    """
    batch_size, seq_len = x.shape[0], x.shape[1]
    context_seq_len = context.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    print(f"Shape of x: {x.shape}, context shape: {context.shape}")
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    context_normed = modulate_fused(self.norm1(context), shift_msa, scale_msa)

    # Compute q from x, k/v from context
    q = self.attn_qkv(x)
    k_v = self.attn_qkv(context_normed)

    # Rearrange for multi-head
    q = rearrange(q, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    k_v = rearrange(k_v, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

    # Only use q from x, k/v from context
    q = q[:, :, 0]  # (b, s, h, d)
    k = k_v[:, :, 1]  # (b, context_seq, h, d)
    v = k_v[:, :, 2]  # (b, context_seq, h, d)

    # Optionally apply rotary to q (and k if needed)
    cos, sin = rotary_cos_sin
    print(f"Calling Cross-attention rotary")
    q = apply_rotary_pos_emb(q, cos[:, :, 0, :, :].to(q.dtype), sin[:, :, 0, :, :].to(q.dtype))
    # Optionally, apply rotary to k as well if needed (not always required for cross-attn)

    # Transpose for attention: (b, h, s, d) and (b, h, context_seq, d)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Cross attention
    x = F.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.max_seq_len = 2 * config.model.max_seq_len if config.model.use_puzzle_conditioning else config.model.max_seq_len
    if self.config.data.padded_vocab:
      self.rounded_vocab_size = vocab_size + (128 - vocab_size % 128) % 128
    else:
      self.rounded_vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.rounded_vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads,
      max_seq_len=self.max_seq_len,
    )
    self.rotary_emb_cross = Rotary(
      config.model.hidden_size // config.model.n_heads,
      max_seq_len=config.model.max_seq_len
    )

    blocks = []
    blocks_cross = []
    # if False:
    if config.model.use_puzzle_conditioning:
      for i in range(config.model.n_blocks):
        blocks.append(DDiTBlock(config.model.hidden_size,
                                    config.model.n_heads,
                                    config.model.cond_dim,
                                    dropout=config.model.dropout))
      blocks_cross.append(DDiTBlockCross(config.model.hidden_size,
                                    config.model.n_heads,
                                    config.model.cond_dim,
                                    dropout=config.model.dropout))
      self.blocks_cross = nn.ModuleList(blocks_cross)
    else:
      for _ in range(config.model.n_blocks):
        blocks.append(DDiTBlock(config.model.hidden_size,
                                config.model.n_heads,
                                config.model.cond_dim,
                                dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      self.rounded_vocab_size,
      config.model.cond_dim)
    
    self.register_buffer("logit_bias", torch.full((1, 1, 1), 0.0))

  def flops(self, seq_len=128):
    return sum(b.flops(seq_len) for b in self.blocks)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)
    rotary_cos_sin_cross = self.rotary_emb_cross(x)

    # if False:
    if self.config.model.use_puzzle_conditioning:
      for i in range(len(self.blocks) // 2):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      print(f"Calling forward on cross attention block with shape: {x.shape}, rotary_cos_sin shape: {rotary_cos_sin_cross[0].shape}, c shape: {c.shape}, max_seq_len: {self.config.model.max_seq_len}")
      x = self.blocks_cross[0](x[:, -self.config.model.max_seq_len:], x[:, :-self.config.model.max_seq_len], rotary_cos_sin_cross, c, seqlens=None)
      # TODO: cross attention block outputs a seuence of length 81, but self attention works on sequences of length 162!
      # -> Fix!
      for i in range(len(self.blocks) // 2, len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    x = self.output_layer(x, c)

    x = x.scatter_add(-1, indices.unsqueeze(-1), self.logit_bias.to(x.dtype).expand_as(x))

    return x
