import numpy as np

def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m,n))
    Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0], 2)
    for i in range(1, n):
        Q[:, i] = A[:, i]
        for j in range(0, i):
            inner = np.dot(Q[:, j].T, Q[:, i])
            Q[:, i] = Q[:, i] - np.dot(inner, Q[:, j])
        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i], 2)
    return Q

def qr_gs(A):
    m, n = A.shape
    Q = gram_schmidt(A)
    R = np.zeros((m,n))
    for i in range(0, n):
        R[i, i] = np.dot(Q[:, i], A[:, i])
        for j in range(0, i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
    return Q, R

if __name__ == "__main__":
    A = np.array([1])