# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_3d_blocks.py

import torch
from torch import nn
from models.resnet import Downsample, ResnetBlock, Upsample
from models.transformers import SpatialTransformer, TemporalTransformer, SpatioTemporalTransformer

def get_down_block(
    down_block_type,
    num_layers,
    transformer_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attention_head_dim,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            transformer_layers=transformer_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    transformer_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attention_head_dim,
    resnet_groups=None,
    cross_attention_dim=None,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            transformer_layers=transformer_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1024,
        upcast_attention: bool = False
    ):
        super().__init__()

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        
        attentions = []
        spatio_temp_attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                SpatialTransformer(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    upcast_attention=upcast_attention
                )
            )
            spatio_temp_attentions.append(
                SpatioTemporalTransformer(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups
                )
            )
            temp_attentions.append(
                TemporalTransformer(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups
                )
            )
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.spatio_temp_attentions = nn.ModuleList(spatio_temp_attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        
    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None
    ):
        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.resnets[0]), hidden_states, temb)
        else:
            hidden_states = self.resnets[0](hidden_states, temb)
            
        for attn, spatio_temp_attn, temp_attn, resnet in zip(self.attentions, self.spatio_temp_attentions, self.temp_attentions, self.resnets[1:]):
            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(spatio_temp_attn, return_dict=False), hidden_states)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = spatio_temp_attn(hidden_states).sample
                hidden_states = temp_attn(hidden_states).sample
                hidden_states = resnet(hidden_states, temb)

        return hidden_states

class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample:bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False
    ):
        super().__init__()

        resnets = []
        attentions = []
        temp_attentions = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention
                )
            )
            temp_attentions.append(
                TemporalTransformer(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None
    ):
        output_states = ()

        for resnet, attn, temp_attn in zip(self.resnets, self.attentions, self.temp_attentions):
            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = temp_attn(hidden_states).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample:bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()

        resnets = []

        self.gradient_checkpointing = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False
    ):
        super().__init__()

        resnets = []
        attentions = []
        temp_attentions = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention
                )
            )
            temp_attentions.append(
                TemporalTransformer(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None
    ):
        for resnet, attn, temp_attn in zip(self.resnets, self.attentions, self.temp_attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = temp_attn(hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True
    ):
        super().__init__()
        resnets = []

        self.gradient_checkpointing = False
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states