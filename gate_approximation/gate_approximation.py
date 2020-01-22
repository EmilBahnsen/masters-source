from diamond import *
from scipy.optimize import minimize
from scipy.optimize import minimize, rosen, rosen_der
from autograd import grad, jacobian
import random
from scipy.optimize import basinhopping
import numba
from numba import jit, njit

circuit = DiamondCircuit(4)
circuit.add_gate('X', [0])

matrix = qc2matrix(circuit)


def print_sparse(matrix: Qobj, atol = 1e-8):
    rows, cols = matrix.shape
    array = matrix.data.asformat('array')
    new_array = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if np.linalg.norm(array[i][j]) > atol:
                new_array[i][j] = 1
    print(new_array)


N = 4  # For now


def system_matrix(qubit_rotations: [[[float]]], u_rotations: [float]):
    """
    Matrix for a system of the kind:
    q1 - Rx Ry Rz - |---| - Rx Ry Rz - |---| - ... - |---| - Rx Ry Rz - |---| - Rx Ry Rz -
    q2 - Rx Ry Rz - |   | - Rx Ry Rz - |   | - ... - |   | - Rx Ry Rz - |   | - Rx Ry Rz -
    q3 - Rx Ry Rz - | U | - Rx Ry Rz - | U | - ... - | U | - Rx Ry Rz - | U | - Rx Ry Rz -
    q4 - Rx Ry Rz - |___| - Rx Ry Rz - |___| - ... - |___| - Rx Ry Rz - |___| - Rx Ry Rz -
    :param qubit_rotations:
    :param u_rotations:
    :return:
    """
    matrix_rotation_block = lambda block: tensor(*[u3(*t) for t in block])
    matrix = I4
    # Apply to the first (rotation, u-pairs)
    for rotation_block, angle_u in zip(qubit_rotations[:-1], u_rotations):
        matrix = matrix_rotation_block(rotation_block) * matrix
        matrix = U(angle_u) * matrix
    # Make the last rotation block
    matrix = matrix_rotation_block(qubit_rotations[-1]) * matrix
    return matrix


r = lambda: random.random() * 2*π

nU = 5
guess = np.array([
    [
        [
            [r(),r(),r()],
            [r(),r(),r()],
            [r(),r(),r()],
            [r(),r(),r()]
        ]
    ] * (nU+1),
    [r()] * nU
])


# CNOT_23 = U2(π) * X(N,1) * X(N,0) * H(N,2) * U2(π) * H(N,3) * X(N,1) * X(N,0)
guess_close_cnot23 = np.array([
    [
        [
            [π, 0, π],
            [π, 0, π],
            [0, 0, 0],
            [π/2, 0, π]
        ], [
            [π, 0, π],
            [π, 0, π],
            [π/2, 0, π],
            [0, 0, 0]
        ], [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ],
    [π, π]
])



# CNOT on T1,T2
target = cnot(N, 2, 3)


def pack_angles(thetas):
    n_U = (len(thetas) - 12) // 13  # Original guess is 4·3·(m-1) + m = 13m + 12
    return [
        np.reshape(thetas[:-n_U], (n_U + 1, 4, 3)),
        thetas[-n_U:],
    ]



def loss(thetas: [], target = target):
    matrix = system_matrix(*pack_angles(thetas))
    diff = target - matrix
    return sqrt((diff.dag() * diff).tr())  # Frobenius norm




# loss_jacobian = nd.Jacobian(loss)
# print(loss_jacobian([1]*30))


def flatten(data, acc = None):
    if not isinstance(acc, np.ndarray):
        acc = np.array([])
    for d in data:
        if isinstance(d, list):
            acc = flatten(d, acc)
        else:
            acc = np.append(acc, d)
    return acc


_target = U(π) * X(N,1) * X(N,0) * H(N,2) * U(π) * H(N,3) * X(N,1) * X(N,0)
# _guess = system_matrix(*guess_close_cnot23)
# print('target, _target, _guess')
# print_sparse(target)
# print_sparse(_target)
# print_sparse(_guess)
# print(_guess - _target)
# print(loss(flatten(guess_close_cnot23), _target))
# exit()

vector_guess = flatten(guess)

# result = minimize(
#     loss,
#     vector_guess,
#     bounds=[(0, 2*π)]*len(vector_guess),
#     method='SLSQP',
#     options={
#         'maxiter': 10000,
#         'ftol': 1e-06,
#         'iprint': 1,
#         'disp': False,
#         'eps': 1.4901161193847656e-08
#     }
# )

class PeriodicBounds(object):
    def __init__(self, xmin, xmax):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


bounds = PeriodicBounds([0] * len(vector_guess), [2*π] * len(vector_guess))


def print_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))
    print(x)
    return f < 0.01


result = basinhopping(
    lambda thetas: loss(thetas, target),
    vector_guess,
    T=1,
    interval=10,
    minimizer_kwargs={'method': 'BFGS'},
    niter=200,
    # accept_test=bounds,
    callback = print_fun,
    stepsize=2*π/3,
    disp=True
)

print(pack_angles(result))