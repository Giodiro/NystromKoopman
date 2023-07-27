from typing import Optional, Callable

import torch


class ConjugateGradient:
    def __init__(self):
        self.num_iter = None

    def solve(self,
              X0: Optional[torch.Tensor],
              B: torch.Tensor,
              mmv: Callable[[torch.Tensor], torch.Tensor],
              max_iter: int) -> torch.Tensor:
        if X0 is None:
            R = torch.clone(B, memory_format=torch.preserve_format)  # n*t
            X = torch.clone(B, memory_format=torch.preserve_format)  # n*t
            X.fill_(0.0)
        else:
            R = B - mmv(X0)  # n*t
            X = X0

        m_eps = 1e-5
        full_grad_every = 10
        tol = 1e-4 ** 2

        P = R.clone()
        Rsold = R.square().sum(dim=0)

        for self.num_iter in range(max_iter):
            AP = mmv(P)
            alpha = Rsold / (torch.sum(P * AP, dim=0).add_(m_eps))
            # X += P @ diag(alpha)
            X.addcmul_(P, alpha.reshape(1, -1))

            if (self.num_iter + 1) % full_grad_every == 0:
                if X.is_cuda:
                    # addmm_ may not be finished yet causing mmv to get stale inputs.
                    torch.cuda.synchronize()
                R = B - mmv(X)
            else:
                # R -= AP @ diag(alpha)
                R.addcmul_(AP, alpha.reshape(1, -1), value=-1.0)

            Rsnew = R.square().sum(dim=0)  # t
            converged = torch.less(Rsnew, tol)
            if torch.all(converged):
                break

            # P = R + P @ diag(mul)
            multiplier = (Rsnew / Rsold.add_(m_eps)).reshape(1, -1)
            P = P.mul_(multiplier).add_(R)
            Rsold = Rsnew
        return X
