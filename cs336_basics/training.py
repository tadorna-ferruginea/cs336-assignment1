from jaxtyping import Float, Int
from torch import Tensor

def avg_cross_entropy(
    inputs: Float[Tensor, "*batch_size vocab_size"],
    targets: Int[Tensor, "*batch_size"],
) -> Float[Tensor, ""]:
    log_probs = inputs.log_softmax(dim=-1)
    return -log_probs.take_along_dim(targets.unsqueeze(-1), dim=-1).squeeze(-1).mean()