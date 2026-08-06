import math
import torch
import torch.nn as nn
from torch import Tensor
import einx
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:

        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(
            torch.empty(
                d_out,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Float[Tensor, "*batch in_features"]) -> Float[Tensor, "*batch out_features"]:

        return einx.dot("out_features [in_features], ... [in_features] -> ... out_features", self.weight, x)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch embedding_dim"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: Float[Tensor, "*batch in_features"]) -> Float[Tensor, "*batch in_features"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        frac_x_rms = x * torch.rsqrt(einx.mean("... ([in_features])", x.pow(2)) + self.eps)
        frac_x_rms = frac_x_rms.to(in_dtype)
        return einx.multiply("... in_features, in_features -> ... in_features", frac_x_rms, self.weight)


class SwiGLU(nn.Module):
    """
    W1 goes through SiLU,
    W2 bypass,
    W1 W2 bind together to go through W3
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        obtain d_ff, around 8/3 d_model and round up to multiples of 64
        initialize W1, W2 and W3 with our previous Linear Module
        """
        super().__init__()
        if not d_ff:
            d_ff = d_model * 8 // 3
            d_ff = ((d_ff - 1) // 64 + 1) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "*batch in_features"]) -> Float[Tensor, "*batch out_features"]:
        """
        1. x -w1-> x1 -SiLU-> x2
        2. x -w3-> x3
        3. x2, x3 -bind-> x4 -w2-> output
        """
        # path 1:
        x1 = self.w1(x)
        x2 = x1 * torch.sigmoid(x1)

        # path 2:
        x3 = self.w3(x)

        # path 3
        x4 = x2 * x3
        output = self.w2(x4)

        return output
