import math
import torch
import torch.nn as nn
from torch import Tensor
import einx
from jaxtyping import Float, Int


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

    def forward(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        frac_x_rms = x * torch.rsqrt(einx.mean("... ([in_features])", x.pow(2)) + self.eps)
        frac_x_rms = frac_x_rms.to(in_dtype)
        return einx.multiply("... in_features, in_features -> ... in_features", frac_x_rms, self.weight)


class SwiGLU(nn.Module):
    """
    W1 goes through SiLU,
    W3 bypass,
    W1 W3 bind together to go through W2
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

    def forward(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
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


class RoPE(nn.Module):
    sin_cached: Tensor
    cos_cached: Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert d_k % 2 == 0, "d_k need to be even in RoPE"

        i_seq = torch.arange(0, max_seq_len, dtype=torch.float32)
        k_seq = torch.pow(theta, -torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        # get angles
        angles = torch.outer(i_seq, k_seq)

        # cache the sin and cos
        # buffer is a dict like parameter
        # and since this is not obtained from training
        # we put persistent to false to avoid adding to state_dict
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)

    def forward(
        self, x: Float[Tensor, "*batch seq_len d_k"], token_positions: Int[Tensor, "*batch seq_len"]
    ) -> Float[Tensor, "*batch seq_len d_k"]:
        # devide x into 2 sub lattices
        x0_even, x0_odd = einx.id("... seq (k (1+1)) -> ... seq k, ... seq k", x)

        # we notice that token_positions is a collection of all the "i"s
        sin = einx.get_at("[i] k, ... seq -> ... seq k", self.sin_cached, token_positions)
        cos = einx.get_at("[i] k, ... seq -> ... seq k", self.cos_cached, token_positions)

        # let's rock and RoPE
        x1_even = x0_even * cos - x0_odd * sin
        x1_odd = x0_even * sin + x0_odd * cos

        x_out = einx.id("... seq k, ... seq k -> ... seq (k (1+1))", x1_even, x1_odd)
        return x_out
