# import numpy as np

# case 1
# x = A[:, 0]
# xnorm = np.linalg.norm(x)
# sign = np.sign(x[0])
# y = np.zeros_like(x)
# y[0] = -sign * xnorm

# w = (x-y) / np.linalg.norm(x-y)
# P = np.eye(len(x)) - 2 * np.outer(w, w)

# case 2 (general)
# leaves the first k-1 components of x unchanged and k - n are zero

import numpy as np

def householder_case2_for_vector(x, k):
    """
    Case II: Build P = I - 2 w w^T such that:
      - (Px)_1..(Px)_{k-1} = x_1..x_{k-1}  (unchanged)
      - (Px)_k = -sign(x_k) * ||x_k:||
      - (Px)_{k+1..n} = 0
    """
    x = x.astype(float)
    n = x.size

    # 1) Tail and its norm
    tail = x[k:]
    r = np.linalg.norm(tail)
    if r == 0:
        return np.eye(n)  # nothing to do

    # 2) Stable sign
    s = -1.0 if tail[0] >= 0 else 1.0

    # 3) Build target y
    y = x.copy()
    y[k] = s * r
    y[k+1:] = 0.0

    # 4) Form u, then w
    u = x - y
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.eye(n)
    w = u / u_norm

    # 5) Full Householder
    P = np.eye(n) - 2.0 * np.outer(w, w)
    return P



if __name__ == "__main__":

    A = np.array([
        [4, 1, -2, 2],
        [1, 2, 0, 1],
        [-2, 0, 3, -2],
        [2, 1, -2, -1]
    ], dtype=float)

    x = A[:, 0]
    P1 = householder_case2_for_vector(x, k=1)
    print(P1)

    A2 = P1 @ A @ P1
    print(A2)

    x = A2[:, 1]
    P2 = householder_case2_for_vector(x, k=2)
    print(P2)

    A3 = P2 @ P1 @ A @ P1 @ P2
    print(A3)

    print(np.linalg.eig(A)[0])
    print(np.linalg.eig(A3)[0]) # they're the same!!!
    print("------------------------------")
    A = np.array([
        [-1, 0],
        [2, 10],
        [2, 11]
    ])
    x = A[:, 0]
    P1 = householder_case2_for_vector(x, k=0)
    print(P1)

    A2 = P1 @ A
    x = A2[:, 1]
    P2 = householder_case2_for_vector(x, k=1)
    print(P2)

    print(P2 @ P1 @ A) # <- this is an upper triangular matrix!

