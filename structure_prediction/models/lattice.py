import paddle
from paddle import linalg as LA
import numpy as np

class CrystalFamily(paddle.nn.Layer):

    def __init__(self):
        
        super(CrystalFamily, self).__init__()
        
        basis = self.get_basis()
        masks, biass = self.get_spacegroup_constraints()
        family = self.get_family_idx()
        
        self.register_buffer(name='basis', tensor=basis)
        self.register_buffer(name='masks', tensor=masks)
        self.register_buffer(name='biass', tensor=biass)
        self.register_buffer(name='family', tensor=family)

    def get_basis(self):
        
        basis = paddle.to_tensor(data=[
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], 
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], 
            [[1.0, 0.0,0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], 
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ], dtype='float32')
        
        # Normalize
        basis = basis / basis.norm(axis=(-1, -2)).unsqueeze(axis=-1).unsqueeze(axis=-1)
        
        return basis

    def get_spacegroup_constraint(self, spacegroup):
        
        mask = paddle.ones(shape=[6])
        bias = paddle.zeros(shape=[6])
        if 195 <= spacegroup <= 230:
            pos = [0, 1, 2, 3, 4]
            mask[pos] = 0.0
        elif 143 <= spacegroup <= 194:
            pos = [0, 1, 2, 3]
            mask[pos] = 0.0
            bias[0] = -0.25 * np.log(3) * np.sqrt(2)
        elif 75 <= spacegroup <= 142:
            pos = [0, 1, 2, 3]
            mask[pos] = 0.0
        elif 16 <= spacegroup <= 74:
            pos = [0, 1, 2]
            mask[pos] = 0.0
        elif 3 <= spacegroup <= 15:
            pos = [0, 2]
            mask[pos] = 0.0
        elif 0 <= spacegroup <= 2:
            pass
        return mask, bias

    def get_spacegroup_constraints(self):
        masks, biass = [], []
        for i in range(231):
            mask, bias = self.get_spacegroup_constraint(i)
            masks.append(mask.unsqueeze(axis=0))
            biass.append(bias.unsqueeze(axis=0))
        return paddle.concat(x=masks, axis=0), paddle.concat(x=biass, axis=0)

    def get_family_idx(self):
        family = []
        for spacegroup in range(231):
            if 195 <= spacegroup <= 230:
                family.append(6)
            elif 143 <= spacegroup <= 194:
                family.append(5)
            elif 75 <= spacegroup <= 142:
                family.append(4)
            elif 16 <= spacegroup <= 74:
                family.append(3)
            elif 3 <= spacegroup <= 15:
                family.append(2)
            elif 0 <= spacegroup <= 2:
                family.append(1)
        return paddle.to_tensor(data=family, dtype='int64')

    def de_so3(self, L):
        x = L
        perm_6 = list(range(x.ndim))
        perm_6[-1] = -2
        perm_6[-2] = -1
        LLT = L @ x.transpose(perm=perm_6)
        L_sym = sqrtm(LLT)
        return L_sym

    def v2m(self, vec):
        batch_size, dims = tuple(vec.shape)
        if dims == 6:
            basis = self.basis
        elif dims == 5:
            basis = self.basis[:-1]
        log_mat = paddle.einsum('bk, kij -> bij', vec, basis)
        mat = expm(log_mat)
        return mat

    def m2v(self, mat):
        log_mat = logm(mat)
        vec = paddle.einsum('bij, kij -> bk', log_mat, self.basis)
        return vec

    def proj_k_to_spacegroup(self, vec, spacegroup):
        batch_size, dims = tuple(vec.shape)
        if dims == 6:
            masks = self.masks[spacegroup, :]
            biass = self.biass[spacegroup, :]
        elif dims == 5:
            masks = self.masks[spacegroup, :-1]
            biass = self.biass[spacegroup, :-1]
        return vec * masks + biass

def logm(A):
    det = LA.det(x=A)
    mask = ~(det > 0)
    b = mask.sum()
    if b > 0:
        A[mask] = paddle.eye(num_rows=3).unsqueeze(axis=0).to(A).expand(shape=[b, -1, -1])
    
    eigenvalues, eigenvectors = LA.eig(x=A)

    return paddle.to_tensor(np.real(np.einsum('bij,bj,bjk->bik', eigenvectors.numpy(), np.log(eigenvalues.numpy()), np.linalg.inv(eigenvectors.numpy()))))

def expm(A):
    if isinstance(A, paddle.Tensor):
        return LA.matrix_exp(A)


def sqrtm(A):
    det = LA.det(x=A)
    mask = ~(det > 0)
    b = mask.sum()
    if b > 0:
        A[mask] = paddle.eye(num_rows=3).unsqueeze(axis=0).to(A).expand(shape=[b, -1, -1])
    
    eigenvalues, eigenvectors = LA.eig(x=A)
    eigenvalues_real = paddle.real(eigenvalues)
    eigenvectors_real = paddle.real(eigenvectors)
    
    eigenvectors_inv = paddle.linalg.pinv(eigenvectors_real)
    
    return paddle.einsum('bij,bj,bjk->bik', eigenvectors_real, eigenvalues_real.sqrt(), eigenvectors_inv)