import paddle
from . import soft_dtw  
from . import path_soft_dtw  

def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0: 2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply  
    D = paddle.zeros((batch_size, N_output, N_output))
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k,:,:].reshape(-1,1), outputs[k,:,:].reshape(-1,1))
        D[k:k+1,:,:] = Dk     
    loss_shape = softdtw_batch(D, gamma)
    
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)           

    Omega = soft_dtw.pairwise_distances(
        paddle.arange(1.0, float(N_output+1)).reshape(N_output,1)
    )
    loss_temporal =  paddle.sum(path * Omega) / (N_output * N_output) 
    
    loss = alpha*loss_shape + (1-alpha)*loss_temporal
    return loss