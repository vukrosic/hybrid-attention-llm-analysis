import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
from torchtune.modules import RotaryPositionalEmbeddings

# Import the high-performance kernels
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 1000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    # Hybrid attention parameters
    layer_types: Optional[List[str]] = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 8
    linear_num_value_heads: int = 8

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Set default layer types if not provided (every 4th layer uses full attention)
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % 4 != 0 else "full_attention"
                for i in range(self.n_layers)
            ]

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: MoEModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x, cache=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, H, T, D]

        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        # Handle KV cache for generation
        if cache is not None:
            # Concatenate with past keys and values
            K = torch.cat([cache['key'], K], dim=2)
            V = torch.cat([cache['value'], V], dim=2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Prepare cache for next step
        new_cache = {'key': K, 'value': V}
        
        return self.w_o(attn_output), new_cache


class GatedDeltaNet(nn.Module):
    """GatedDeltaNet layer using high-performance fla and causal-conv1d kernels"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        # Calculate projection dimensions
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = 'silu'

        # 1. Input Projections
        proj_qkvz_size = self.key_dim * 2 + self.value_dim * 2
        proj_ba_size = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.d_model, proj_qkvz_size, bias=False)
        self.in_proj_ba = nn.Linear(self.d_model, proj_ba_size, bias=False)
        
        # 2. Convolution Layer
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            padding=config.linear_conv_kernel_dim - 1,
        )

        # 3. Discretization Parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # 4. Fused Normalization and Gating
        self.norm = FusedRMSNormGated(self.value_dim, activation=self.activation)
        
        # 5. Final Output Projection
        self.out_proj = nn.Linear(self.value_dim, self.d_model, bias=False)

    def forward(self, x, cache=None):
        batch_size, seq_len, _ = x.shape
        is_inference = seq_len == 1 and cache is not None

        # =================================================================
        # 1. Projections and Gate Calculations
        # =================================================================
        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)
        
        query, key, value, z = torch.split(qkvz, [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1)
        beta, alpha = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)

        # Calculate recurrent gates
        beta = beta.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
        
        # =================================================================
        # 2. Causal Convolution Step
        # =================================================================
        qkv_conv_input = torch.cat([query, key, value], dim=-1).transpose(1, 2)
        
        if is_inference:
            # INFERENCE PATH: Use the _update kernel for a single token
            qkv_conv_output = causal_conv1d_update(
                qkv_conv_input, cache['conv_state'],
                self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
            )
        else:
            # TRAINING/PREFILL PATH: Use the _fn kernel for the whole sequence
            qkv_conv_output = causal_conv1d_fn(
                x=qkv_conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation
            )
        
        qkv_conv_output = qkv_conv_output.transpose(1, 2)
        query, key, value = torch.split(qkv_conv_output, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # =================================================================
        # 3. Reshape and Gated Delta Rule (Recurrent Step)
        # =================================================================
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Repeat K/Q heads if they are grouped with more V heads
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
            
        if is_inference:
            # INFERENCE PATH: Use the fused recurrent kernel for a single step
            core_output, new_recurrent_state = fused_recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=cache['recurrent_state'],
                output_final_state=True
            )
        else:
            # TRAINING/PREFILL PATH: Use the chunked kernel for numerical stability
            core_output, new_recurrent_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=True
            )
        
        # =================================================================
        # 4. Fused Gated Normalization & Final Projection
        # =================================================================
        core_output = core_output.reshape(batch_size * seq_len, self.value_dim)
        z = z.reshape(batch_size * seq_len, self.value_dim)
        
        # This one call does: RMSNorm(core_output) * silu(z)
        fused_output = self.norm(core_output, z)
        fused_output = fused_output.view(batch_size, seq_len, self.value_dim)

        output = self.out_proj(fused_output)

        # For caching, prepare the new states
        new_cache = {
            'conv_state': qkv_conv_input if is_inference else F.pad(qkv_conv_input, (0, 0, self.conv1d.kernel_size[0] - seq_len, 0)),
            'recurrent_state': new_recurrent_state
        }

        return output, new_cache


class DynamicCache:
    """Cache manager for hybrid attention models with both MultiHeadAttention and GatedDeltaNet layers"""
    def __init__(self, config: MoEModelConfig, batch_size: int, device: torch.device):
        self.key_cache = [None] * config.n_layers
        self.value_cache = [None] * config.n_layers
        self.conv_states = [None] * config.n_layers
        self.recurrent_states = [None] * config.n_layers
        self.config = config
        
        # Pre-allocate cache memory for GatedDeltaNet layers
        for i in range(config.n_layers):
            if config.layer_types[i] == "linear_attention":
                # Fixed-size caches for GatedDeltaNet
                conv_dim = (config.linear_num_key_heads * config.linear_key_head_dim) * 2 + (config.linear_num_value_heads * config.linear_value_head_dim)
                self.conv_states[i] = torch.zeros(
                    batch_size, conv_dim, config.linear_conv_kernel_dim, device=device
                )
                self.recurrent_states[i] = torch.zeros(
                    batch_size, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim, device=device
                )

    def get_layer_cache(self, layer_idx: int) -> Optional[dict]:
        """Returns the cache for a specific layer."""
        if self.config.layer_types[layer_idx] == "full_attention":
            if self.key_cache[layer_idx] is None:
                return None
            return {'key': self.key_cache[layer_idx], 'value': self.value_cache[layer_idx]}
        else:  # linear_attention
            return {'conv_state': self.conv_states[layer_idx], 'recurrent_state': self.recurrent_states[layer_idx]}

    def update_layer_cache(self, layer_idx: int, new_cache: dict):
        """Updates the cache for a specific layer."""
        if self.config.layer_types[layer_idx] == "full_attention":
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = new_cache['key']
                self.value_cache[layer_idx] = new_cache['value']
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_cache['key']], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_cache['value']], dim=2)
        else:  # linear_attention
            self.conv_states[layer_idx] = new_cache['conv_state']
            self.recurrent_states[layer_idx] = new_cache['recurrent_state']


class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight

class MoETransformerBlock(nn.Module):
    """Hybrid Transformer block with MoE supporting both full attention and GatedDeltaNet"""
    def __init__(self, config: MoEModelConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        # Choose attention mechanism based on layer type
        if self.layer_type == "full_attention":
            self.token_mixer = MultiHeadAttention(
                config.d_model, config.n_heads, config.max_seq_len, config.dropout
            )
        elif self.layer_type == "linear_attention":
            self.token_mixer = GatedDeltaNet(config)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            config.d_model, config.d_ff, config.num_experts, config.expert_top_k, config.dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None):
        # Token mixing (attention or linear attention)
        mixer_output, new_cache = self.token_mixer(self.norm1(x), cache=cache)
        x = x + self.dropout(mixer_output)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss, new_cache


class MoEMinimalLLM(nn.Module):
    """Minimal LLM with Mixture of Experts"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Hybrid transformer blocks with MoE
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(config, i) for i in range(config.n_layers)
        ])

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, cache: Optional[DynamicCache] = None, return_aux_loss=True):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Collect auxiliary losses from MoE layers
        aux_losses = []
        is_generating = cache is not None

        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            layer_cache = cache.get_layer_cache(i) if is_generating else None
            
            x, aux_loss, new_layer_cache = block(x, cache=layer_cache)
            
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)
            
            # Update cache during generation or evaluation
            if is_generating or not self.training:
                if cache is None and not self.training:  # Create cache on first eval pass
                    cache = DynamicCache(self.config, x.size(0), x.device)
                if cache is not None:
                    cache.update_layer_cache(i, new_layer_cache)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: MoEModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                # MoE model evaluation
                logits = model(x, return_aux_loss=False)  # Don't return aux loss during eval
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: MoEModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]


def train_moe_model(config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the MoE model"""
    print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")

    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training MoE")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            if config.use_amp:
                with autocast():
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                    # Combine main loss and auxiliary loss
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

            # Milestone evaluations
            if step in getattr(config, 'log_milestones', ()):    
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")

            step += 1
            if step % 20 == 0:
                pbar.update(20)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")

    return model, final_eval


def generate(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text using the hybrid model with efficient caching"""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Initialize the cache for the batch
    cache = DynamicCache(model.config, batch_size=input_ids.shape[0], device=device)
    
    # Prefill step (process the prompt)
    with torch.no_grad():
        logits = model(input_ids, cache=cache, return_aux_loss=False)

    # Get the next token
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
    generated_ids = [next_token.item()]
    
    # Decoding loop
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits = model(next_token, cache=cache, return_aux_loss=False)
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_ids.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Load data first to get vocab_size
    temp_config = MoEModelConfig()  # Use MoE config for data loading
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    # Use MoE config and set vocab_size
    config = MoEModelConfig(vocab_size=vocab_size)

    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train MoE model
    print(f"\n{'='*60}")
    print(f"🧪 TRAINING: Mixture of Experts Model")
    print(f"{'='*60}")

    print(f"\n📋 Hybrid MoE Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   MoE: {config.num_experts} experts, top-{config.expert_top_k} routing")
    print(f"   Hybrid: {config.layer_types.count('full_attention')} full attention, {config.layer_types.count('linear_attention')} GatedDeltaNet layers")
    print(f"   GatedDeltaNet: {config.linear_num_key_heads}K/{config.linear_num_value_heads}V heads, {config.linear_key_head_dim}K/{config.linear_value_head_dim}V dims, conv_k={config.linear_conv_kernel_dim}")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Train model
    start_time = time.time()
    model, final_metrics = train_moe_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎯 Hybrid MoE Model Results:")
    print(f"⏱️ Training time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    # Demonstrate generation capabilities
    print(f"\n🤖 Testing Generation with Hybrid Caching:")
    test_prompt = "The future of artificial intelligence"
    generated_text = generate(model, tokenizer, test_prompt, max_new_tokens=20)
    print(f"   Prompt: '{test_prompt}'")
    print(f"   Generated: '{generated_text}'")
    
    print(f"{'='*60}")
    print(f"✅ Successfully implemented hybrid attention with GatedDeltaNet!")
    print(f"📊 Model uses {config.layer_types.count('linear_attention')} linear attention layers for efficiency")
    print(f"🚀 and {config.layer_types.count('full_attention')} full attention layers for expressiveness")
    print(f"{'='*60}")