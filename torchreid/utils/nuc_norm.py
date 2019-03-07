import torch
from torch.autograd import Variable

iter_times = 10


def compute_error(A, B):

    error = A - B
    return torch.sqrt(error * error).sum()


def generate_symm_matrix(batch_size, C):

    A = torch.rand(batch_size, C, C, device='cuda', requires_grad=True)

    return torch.bmm(A.permute(0, 2, 1), A)


def msqrt(A):
    """
    Newton-Schulz Iteration Version.
    Copy from: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py.
    4 times faster than SVD version.
    """
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim, dim, device='cuda').view(1, dim, dim).repeat(batchSize, 1, 1)  # noqa
    Z = torch.eye(dim, dim, device='cuda').view(1, dim, dim).repeat(batchSize, 1, 1)
    for i in range(iter_times):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def _apply_func(func, M):

    tList = [func(m) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)

    return res


def binv(M: 'N x C x C'):

    return _apply_func(torch.inverse, M)


EPSILON = 1e-12  # for numeric stability


class NucNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A: 'N x C x C'):

        N, C, _ = A.size()
        ATA = torch.bmm(A.permute(0, 2, 1), A)
        eye = torch.eye(C, device='cuda').expand(N, C, C)
        masked = msqrt(ATA * eye + EPSILON)
        ctx.save_for_backward(A, masked)
        return torch.sum(masked, dim=(1, 2))

    @staticmethod
    def backward(ctx, grad_output: 'N'):

        N = grad_output.size(0)
        A, masked = ctx.saved_tensors
        C = A.size(1)

        grad_output = grad_output.clone().view(N, 1, 1).repeat(1, C, C)
        grad_norm = torch.bmm(
            A,
            binv(masked)
        )

        return grad_output * grad_norm


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--iters', default=10, type=int)
    parser.add_argument('--epsilon', default=1e-12, type=float)
    options = parser.parse_args()

    iter_times = options.iters
    EPSILON = options.epsilon

    my_nuc_norm = NucNorm.apply
    print('Generating matrix for testing...')
    A = Variable(generate_symm_matrix(options.batch, options.size))
    dt = Variable(torch.rand(options.batch, device='cuda'), requires_grad=False)
    print('Testing msqrt...')
    A_ = A.clone()
    sA_ = msqrt(A)
    print(compute_error(A_, torch.bmm(sA_, sA_)))
    print('Applying torch.norm...')
    A_norm_1 = _apply_func(lambda A: torch.norm(A, p='nuc'), A.clone())
    print(A_norm_1)
    print('Applying custom norm...')
    A_norm_2 = my_nuc_norm(A.clone())
    print(A_norm_2)

    print(compute_error(A_norm_1, A_norm_2))
