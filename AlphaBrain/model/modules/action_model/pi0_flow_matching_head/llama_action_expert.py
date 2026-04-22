"""
Llama Action Expert for LlamaPi0 Architecture

Uses Llama transformer architecture instead of Gemma for the action expert.
Key advantage: attention head format (num_heads=32, num_kv_heads=8, head_dim=128)
matches Llama VLM exactly, enabling true joint attention.

Architecture:
- Based on HuggingFace LlamaModel (not LlamaForCausalLM, no lm_head needed)
- 1024 hidden_size, 18 layers, ~400M parameters
- Compatible with π₀ mode (no adaRMS needed)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from transformers import LlamaConfig, LlamaModel
import logging

logger = logging.getLogger(__name__)


class LlamaActionExpert(nn.Module):
    """
    Independent Llama model that serves as the action expert in LlamaPi0.

    Unlike the Gemma action expert, this uses Llama architecture to match
    the VLM's attention format exactly, enabling true joint attention
    without format conversion overhead.

    Key differences from GemmaActionExpert:
    - Uses LlamaModel instead of GemmaForCausalLM  
    - No adaRMS support (π₀ mode only)
    - Attention heads match Llama VLM: 32 heads, 8 kv_heads, head_dim=128
    - Standard RMSNorm (no adaptive normalization)
    """

    def __init__(
        self,
        width: int = 1024,
        depth: int = 18,
        mlp_dim: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        # Create Llama config matching specifications
        config = LlamaConfig(
            hidden_size=width,
            intermediate_size=mlp_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=32000,  # Standard Llama vocab size (not used since we don't need embeddings)
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            mlp_bias=False,
            # Don't use rope_scaling - will inherit from VLM at runtime
        )

        # Create LlamaModel (base model without lm_head)
        self.model = LlamaModel(config)
        
        # Remove token embeddings since we work with continuous embeddings
        self.model.embed_tokens = None

        # Store configuration
        self.width = width
        self.depth = depth
        self.precision = precision

        # Set precision
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            # Keep norms in float32 for numerical stability
            for name, param in self.named_parameters():
                if "norm" in name.lower():
                    param.data = param.data.to(dtype=torch.float32)

        logger.info(
            f"LlamaActionExpert initialized: {width}d × {depth}L, "
            f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"precision={precision}, params={sum(p.numel() for p in self.parameters())/1e6:.1f}M"
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass through the Llama action expert.
        
        Args:
            inputs_embeds: [B, seq_len, hidden_size] input embeddings
            attention_mask: [B, 1, seq_len, seq_len] attention mask
            position_ids: [B, seq_len] position indices
            past_key_values: optional KV cache from previous forward
            use_cache: whether to return KV cache
            
        Returns:
            tuple: (last_hidden_state, past_key_values)
        """
        output = self.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        return output.last_hidden_state, getattr(output, 'past_key_values', None)

    def get_language_model(self):
        """Return the underlying language model for compatibility."""
        return self.model