import numpy as np
import paddle
from numba import jit
from paddle.autograd import PyLayer

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).reshape([-1, 1])
    if y is not None:
        y_t = paddle.transpose(y, perm=[0, 1])
        y_norm = (y**2).sum(1).reshape([1, -1])
    else:
        y_t = paddle.transpose(x, perm=[0, 1])
        y_norm = x_norm.reshape([1, -1])
    
    dist = x_norm + y_norm - 2.0 * paddle.mm(x, y_t)
    return paddle.clip(dist, 0.0, float('inf'))

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(PyLayer):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
        dev = D.place
        batch_size, N, N = D.shape
        gamma = paddle.to_tensor([gamma], dtype='float32').to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = paddle.zeros((batch_size, N+2, N+2))   
        for k in range(0, batch_size): # loop over all D in the batch    
            Rk = paddle.to_tensor(compute_softdtw(D_[k,:,:], g_), dtype='float32').to(dev)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.place
        D, R, gamma = ctx.saved_tensor()
        batch_size, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = paddle.zeros((batch_size, N, N)) 
        for k in range(batch_size):         
            Ek = paddle.to_tensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_), dtype='float32').to(dev)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None