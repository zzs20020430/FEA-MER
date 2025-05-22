# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                #  d_model=None,        # 适用于 方阵微调
                input_dim,
                output_dim,
                bottleneck=None,        # 实际是rank，也是args.ffn_num
                dropout=0.0,
                init_option="lora",
                adapter_scalar="1.0",
                adapter_layernorm_option="in"):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.down_size = bottleneck

        #before(in) or after(out)
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm = None
        if adapter_layernorm_option == "in":
            self.adapter_layer_norm_before = nn.LayerNorm(input_dim)
        elif adapter_layernorm_option == "out":
            self.adapter_layer_norm = nn.LayerNorm(output_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # self.down_proj = nn.Linear(self.n_embd, 64)
        self.down_proj = nn.Linear(self.input_dim, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.output_dim)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":#初始化lora
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm(up)

        if add_residual:  # False
            output = up + residual
        else:
            output = up
        return output