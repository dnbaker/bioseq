import torch
import torch.nn as nn

class SparseSoftmax(nn.Module):
    """
    SparseSoftmax is a wrapper around entmax's entmax_bisect, which allows us to learn
    the parameter alpha in entmax, which determines how sparse a given layer's output should be
    """
    def __init__(self, alpha_init=1.5, n_iter=24, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 reduction='sum', requires_grad=True):
        super().__init__()
        from entmax import EntmaxBisectLoss
        self.alpha = torch.tensor([alpha_init], dtype=dtype, device=device, requires_grad=requires_grad)
        self.loss = EntmaxBisectLoss(
            self.alpha, n_iter=n_iter, reduction=reduction)

    def forward(self, *args, **kwargs):
        if "target" in kwargs:
            args = args + (kwargs["target"],)
            del kwargs["target"]
        if len(args) == 1:
            from entmax import entmax_bisect
            return entmax_bisect(args[0], self.alpha)
        return self.loss(*args, **kwargs)

Softmax = torch.nn.Softmax

__all__ = ["SparseSoftmax", "Softmax"]
