import math
import torch
import torch.nn as nn
from torch import Tensor
import einx
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "*batch in_features"]) -> Float[Tensor, "*batch out_features"]:

        return einx.dot("out_features [in_features], ... [in_features] -> ... out_features", self.weight, x)
