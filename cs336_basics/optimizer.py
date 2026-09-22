from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.optim import Optimizer


def avg_cross_entropy(
    inputs: Float[Tensor, "*batch_size vocab_size"],
    targets: Int[Tensor, "*batch_size"],
) -> Float[Tensor, ""]:
    log_probs = inputs.log_softmax(dim=-1)
    return -log_probs.take_along_dim(targets.unsqueeze(-1), dim=-1).squeeze(-1).mean()

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)
                    state["variance"] = torch.zeros_like(p.data)

                state["step"] += 1
                state["momentum"] = betas[0] * state["momentum"] + (1 - betas[0]) * grad
                state["variance"] = betas[1] * state["variance"] + (1 - betas[1]) * grad * grad
                m_fxd = state["momentum"]/(1 - betas[0]**state["step"])
                v_fxd = state["variance"]/(1 - betas[1]**state["step"])
                p.data -= lr * (m_fxd / (torch.sqrt(v_fxd) + eps) + weight_decay * p.data)
        return loss