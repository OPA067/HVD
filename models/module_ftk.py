import torch
from torch import nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, p = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, p)).to(dtype=x.dtype, device=x.device) # [b, n, p]
        perturbed_x = x.unsqueeze(1) + noise * sigma # [b, 1, p] + [b, n, p] = [b, n, p]
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices
        indices = torch.sort(indices, dim=-1).values

        perturbed_output = F.one_hot(indices, num_classes=p).float() # [b, n, k, p]
        indicators = perturbed_output.mean(dim=1) # [b, k, p]

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)


f_feat = torch.rand(32, 12, 512)
sorces = torch.rand(32, 12)
model = PerturbedTopK(k=6)
sorces = model(sorces)
f_feat_h = torch.einsum("bkf,bfd->bkd", [sorces, f_feat])
# print(f_feat_h.shape)
