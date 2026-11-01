# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedFluxAttnProcessor2_0,
)
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention_processor import FluxAttnProcessor2_0
# from diffusers.normalization import AdaLayerNorm
# from fluxpaint.attention import FluxCCMAttnProcessor2_0
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import FluxPosEmbed

from diffusers.models.modeling_utils import ModelMixin

import os
import json
from einops import rearrange

from utils.logger import *

logger = get_logger("Transformer2DModel")  # pylint: disable=invalid-name

from icecream import ic



@maybe_allow_in_graph
class BasicMultiViewSingleTransformerBlock(nn.Module):
    def __init__(self, transformer: FluxSingleTransformerBlock, layer_name, use_ma=True, use_ra=False, block_index=0):
        super().__init__()
        self.transformer = transformer
        self.use_ma = use_ma
        self.layer_name = layer_name
        self.use_ra = use_ra
        self.num_in_batch = 4
        self.mva_scale = 1.0
        self.ref_scale = 1.0
        self.block_index = block_index
        self.window_size = 8
        
        processor = FluxAttnProcessor2_0()

    def _initialize_attn_mv_weights(self):
        self.attn_mv.load_state_dict(self.attn.state_dict())
        with torch.no_grad():
            for param in self.mv_out.parameters():
                param.zero_()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        length: int = 1536,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        joint_attention_kwargs = joint_attention_kwargs or {}
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb, # ([1536, 128], [1536, 128])
            **joint_attention_kwargs,
        ) # (4, 1536, 3072)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        

        return hidden_states



@maybe_allow_in_graph
class BasicMultiViewTransformerBlock(nn.Module):
    def __init__(self, transformer: FluxTransformerBlock, layer_name, use_ma=True, use_ra=False):
        super().__init__()
        
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        
        self.num_in_batch = 4
        self.mva_scale = 1.0
        self.ref_scale = 1.0
        
        # TODO: multiview attention

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        # joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs, 
        )

        # if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
        # elif len(attention_outputs) == 3:
        #     attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        # if len(attention_outputs) == 3:
        #     hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class CustomFluxTransformer2DModel(
    ModelMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin
): # ModelMixin for LoRA
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicMultiViewTransformerBlock", "BasicMultiViewSingleTransformerBlock", "FluxTransformerBlock", "FluxSingleTransformerBlock","Sparse3D_Attn"]
    # _skip_layerwise_casting_patterns = ["ccm_embed", "pos_embed", "norm"]

    def __init__(self, transformer: FluxTransformer2DModel):
        super().__init__()
        
        self.transformer = transformer
        
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))
        
        self.init_attention(self.transformer, use_ma=True, use_ra=False)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        strict = kwargs.pop('strict', False)
        init_conv_in = kwargs.pop('init_conv_in', True)
        torch_dtype = kwargs.pop('torch_dtype', torch.bfloat16)
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        is_ckpt = "checkpoint" in pretrained_model_name_or_path
        
        # load sclice index
        index_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.safetensors.index.json')
        single_file_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.safetensors')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        transformer = FluxTransformer2DModel(**config)
        transformer = CustomFluxTransformer2DModel(transformer)
        
        from safetensors.torch import load_file
        
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            shard_files = set(index['weight_map'].values())
            
            transformer_ckpt = {}
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                shard = load_file(shard_path, device='cpu')
                transformer_ckpt.update(shard)
        else:
            transformer_ckpt = load_file(single_file_path, device='cpu')
        
        if strict:
            transformer.load_state_dict(transformer_ckpt, strict=True)
        else:
            missing_keys, unexpected_keys = transformer.load_state_dict(transformer_ckpt, strict=False)
            # Filter out any keys containing "attn_mv" from missing_keys
            missing_keys = [k for k in missing_keys if "lora" not in k]
            if is_ckpt:
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        transformer = transformer.to(dtype=torch_dtype)
        return transformer
    
    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.transformer = self.transformer.to(dtype=dtype)
        if device is not None:
            self.transformer = self.transformer.to(device=device)
        return self

    @property
    def dtype(self):
        """Return the dtype of the transformer."""
        return self.transformer.dtype
    
    
    # def init_condition_conv(self):
    #     self.conv_in = nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1, bias=True, dtype=self.transformer.dtype)
        
    def init_attention(self, transformer, use_ma=True, use_ra=False):
        for double_block_i, double_block in enumerate(transformer.transformer_blocks):
            if isinstance(double_block, FluxTransformerBlock):
                transformer.transformer_blocks[double_block_i] = BasicMultiViewTransformerBlock(
                    double_block,
                    f'transformer_{double_block_i}',
                    use_ma, use_ra
                )

        for single_block_i, single_block in enumerate(transformer.single_transformer_blocks):
            if isinstance(single_block, FluxSingleTransformerBlock):
                if single_block_i >= 19: # [0, 18], [19, 37]
                    use_ma = True
                else:
                    use_ma = False
                transformer.single_transformer_blocks[single_block_i] = BasicMultiViewSingleTransformerBlock(
                    single_block,
                    f'single_{single_block_i}',
                    use_ma, use_ra,
                    single_block_i
                )
        
        
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    # https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_flux.py
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # hidden_states, normal_states, ccm_states = hidden_states.chunk(3, dim=-1)
        # * NEW: condition embeddings
        hidden_states = self.x_embedder(hidden_states)
        # normal_states = self.normal_embedder(normal_states)
        # ccm_states = self.ccm_embedder(ccm_states)
        # hidden_states = hidden_states + normal_states + ccm_states

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]




        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)



        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # # controlnet residual
            # if controlnet_block_samples is not None:
            #     interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            #     interval_control = int(np.ceil(interval_control))
            #     # For Xlabs ControlNet.
            #     if controlnet_blocks_repeat:
            #         hidden_states = (
            #             hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
            #         )
            #     else:
            #         hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    encoder_hidden_states.shape[1],
                    image_rotary_emb,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    length=encoder_hidden_states.shape[1],
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # # controlnet residual
            # if controlnet_single_block_samples is not None:
            #     interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            #     interval_control = int(np.ceil(interval_control))
            #     hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
            #         hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            #         + controlnet_single_block_samples[index_block // interval_control]
            #     )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]


        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
