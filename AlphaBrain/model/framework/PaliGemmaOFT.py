# Copyright 2025 VLA-Engine. All rights reserved.
# PaliGemma-OFT Framework — PaliGemma 3B + action token regression

"""
PaliGemma-OFT Framework

Uses PaliGemma (SigLIP + Gemma 2B) as VLM backbone with action special token
for continuous action prediction via L1 regression.
Mirrors LlamaOFT / QwenOFT architecture but with PaliGemma backbone.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import logging
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm.paligemma_oft import _PaliGemma_OFT_VL_Interface
from AlphaBrain.model.modules.action_model.mlp_action_header import get_action_model
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("PaliGemmaOFT")
class PaliGemma_OFT(BaseFramework):
    """
    PaliGemma + action token OFT framework.
    Predicts continuous actions via L1 regression on action token hidden states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # Use PaliGemmaOFT-specific VLM interface (not get_vlm_model which routes to Pi0 version)
        self.paligemma_vl_interface = _PaliGemma_OFT_VL_Interface(config=self.config)

        # Align action hidden dim with LLM hidden size (Gemma 2B: 2048)
        config.framework.action_model.action_hidden_dim = self.paligemma_vl_interface.model.config.text_config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.paligemma_vl_interface.model, 'gradient_checkpointing_enable'):
            self.paligemma_vl_interface.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for PaliGemma model")

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = "🔍"
        self.action_token_id = self.paligemma_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        self.l1_loss = nn.L1Loss()

    def _build_structured_prompt(self, instruction):
        """Build prompt for PaliGemmaOFT inference/training."""
        action_tokens = self.action_token * self.chunk_len
        return f"{instruction} Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        """Training forward: L1 regression on action tokens."""
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # Structured prompt for multi-task instruction routing
        instructions = [self._build_structured_prompt(inst) for inst in instructions]

        # Build PaliGemma inputs
        paligemma_inputs = self.paligemma_vl_interface.build_paligemma_inputs(
            images=batch_images, instructions=instructions
        )

        # Move to device
        device = next(self.parameters()).device
        paligemma_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in paligemma_inputs.items()}

        outputs = self.paligemma_vl_interface(
            **paligemma_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = paligemma_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List = None,
        instructions: List[str] = None,
        examples: List[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        """Inference: predict normalized actions."""
        if examples is not None:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]
        else:
            batch_images = [to_pil_preserve(imgs) for imgs in batch_images]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        instructions = [self._build_structured_prompt(inst) for inst in instructions]

        paligemma_inputs = self.paligemma_vl_interface.build_paligemma_inputs(
            images=batch_images, instructions=instructions
        )

        device = next(self.parameters()).device
        paligemma_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in paligemma_inputs.items()}

        outputs = self.paligemma_vl_interface(
            **paligemma_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = paligemma_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        """Extract action token embeddings — same logic as QwenOFT/LlamaOFT."""
        if action_token_id is None:
            raise ValueError("action_token_id cannot be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Action tokens insufficient for chunk_len {self.chunk_len}: samples {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries
