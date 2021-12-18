import numpy as np


def lu_decomposition_with_pp(matrix):
    # Get the number of rows
    n = matrix.shape[0]

    tmp_L = np.eye(n, dtype=np.double)
    tmp_U = matrix.copy()
    for i in range(n):
        result = np.where(np.abs(tmp_U[i:, i]) == np.amax(np.abs(tmp_U[i:, i])))
        swap_i = result[0][0]
        tmp_U[[swap_i + i, i], i:n] = tmp_U[[i, swap_i + i], i:n]

        factors = tmp_U[i + 1:, i] / tmp_U[i, i]
        tmp_L[i + 1:, i] = factors
        tmp_U[i + 1:] -= factors[:, np.newaxis] * tmp_U[i]
    return tmp_L, tmp_U


def forward_substitution(tmp_l, tmp_b):
    # get size of L matrix
    n = tmp_l.shape[0]

    # initialize empty array with size b
    tmp_y = np.zeros_like(tmp_b, dtype=np.double)

    # computing tmp_y values
    tmp_y[0] = tmp_b[0] / tmp_l[0, 0]
    for i in range(1, n):
        tmp_y[i] = (tmp_b[i] - np.dot(tmp_l[i, :i], tmp_y[:i])) / tmp_l[i, i]
    return tmp_y


def backward_substitution(tmp_u, tmp_y):
    # get size of U matrix
    n = tmp_u.shape[0]

    # initialize emtpy array with size y
    tmp_x = np.zeros_like(tmp_y, dtype=np.double)

    # computing tmp_x values
    tmp_x[-1] = tmp_y[-1] / tmp_u[-1, -1]
    for i in range(n - 2, -1, -1):
        tmp_x[i] = (tmp_y[i] - np.dot(tmp_u[i, i + 1:], tmp_x[i + 1:])) / tmp_u[i, i]
    return tmp_x


def solve_lin_system_with_lu(mat, x):
    TMP_L, TMP_U = lu_decomposition_with_pp(mat)
    y = forward_substitution(TMP_L, x)
    return backward_substitution(TMP_U, y)


def iterative_ref(A, b, tolerance=1e-10, iterations=1000):
    x = np.ones_like(b, dtype=np.double)
    for i in range(iterations):
        x0 = x.copy()
        r = b - np.dot(A, x0)
        c = np.linalg.solve(A, r)
        x = np.add(x0, c)
        if np.linalg.norm(x - x0, 2) / np.linalg.norm(x0, 2) < tolerance:
            break
    return x


#  correct answer is given by gepp
gepp = np.array([[-5.93025381, 0.10179665, -5.63806883],
                 [5.71248256, -5.33907931, 0.04770739],
                 [5.09042443, 4.20346378, -7.43808245]])
gepp_b = np.array([3.00764463,
                   3.39289929,
                   -3.35276915])

print("correct answer is given by gepp method")
print("built in function")
print(np.linalg.solve(gepp, gepp_b))
print("iterative refinement method")
print(iterative_ref(gepp, gepp_b))
print("GEPP method")
print(solve_lin_system_with_lu(gepp, gepp_b), "\n\n\n")

#  gepp fails and works only with iterative refinement
gepp = np.array([[-5e-3, 1.0, 2.0],
                 [-2.0, -1, 1.0],
                 [-5.0, 5.0, 1]])
gepp_b = np.array([6.0,
                   -9.0,
                   2.0])
print("gepp fails and gives accurate results only with iterative refinement")
print("built in function")
print(np.linalg.solve(gepp, gepp_b))
print("iterative refinement method")
print(iterative_ref(gepp, gepp_b))
print("GEPP method")
print(solve_lin_system_with_lu(gepp, gepp_b), "\n\n\n")
