import torch
from functorch import vmap, grad
from torch.autograd import Function

sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))



class TopK2(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts(xs, k)
       # topk,indices = torch.topk(xs,k)
        v,i = torch.topk(xs,k,dim=1)
        m = torch.zeros_like(xs)
        for idx in range(m.shape[0]):
            m[idx,i[idx]] = 1.0
        ctx.save_for_backward(xs, ts, ps)
        return m #ps

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts, ps = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = sigmoid_grad(xs + ts)
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
       # uv = ps * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

class TopK3(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts_v3(xs, k)
       # topk,indices = torch.topk(xs,k)
        v,i = torch.topk(xs,k,dim=1)
        m = torch.zeros_like(xs)
        for idx in range(m.shape[0]):
            m[idx,i[idx]] = 1.0
        ctx.save_for_backward(xs, ts)
        return m

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = 100*sigmoid_grad(100*(xs + ts))
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
       # uv = ps * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

class TopK3_1(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts_v3(xs, k)
       # topk,indices = torch.topk(xs,k)
        v,i = torch.topk(xs,k,dim=1)
        m = torch.zeros_like(xs)
        for idx in range(m.shape[0]):
            m[idx,i[idx]] = 1.0
        ctx.save_for_backward(xs, ts)
        return m

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = 1.*sigmoid_grad(10*(xs + ts))
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
       # uv = ps * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None




@torch.no_grad()
def _find_ts(xs, k):
    b, n = xs.shape
    assert 0 < k < n
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid((xs + mid)).sum(dim=1) < k  #原来没有乘200
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(xs + ts)


@torch.no_grad()
def _find_ts_v3(xs, k):
    b, n = xs.shape
    assert 0 < k < n
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid(100*(xs + mid)).sum(dim=1) < k  
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(100*(xs + ts))


if __name__ == '__main__':
    pass