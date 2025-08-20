import numpy as np

def finite_difference(S0, K, T, r, sigma, Smax=200, M=200, N=2000,
                      method="explicit", option="call"):
    dt = T / N
    dS = Smax / M
    S = np.linspace(0, Smax, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Payoff at maturity
    if option == "call":
        V[:, -1] = np.maximum(S - K, 0)
    else:
        V[:, -1] = np.maximum(K - S, 0)

    # Boundary conditions
    if option == "call":
        V[-1, :] = Smax - K * np.exp(-r * dt * (N - np.arange(N + 1)))
        V[0, :] = 0
    else:
        V[0, :] = K * np.exp(-r * dt * (N - np.arange(N + 1)))
        V[-1, :] = 0

    j = np.arange(0, M + 1)
    a = 0.5 * dt * (sigma**2 * j**2 - r * j)
    b = -dt * (sigma**2 * j**2 + r)
    c = 0.5 * dt * (sigma**2 * j**2 + r * j)

    # Explicit
    if method == "explicit":
        for n in reversed(range(N)):
            for i in range(1, M):
                V[i, n] = a[i] * V[i-1, n+1] + (1+b[i]) * V[i, n+1] + c[i] * V[i+1, n+1]

    # Implicit
    elif method == "implicit":
        A = np.zeros((M-1, M-1))
        for i in range(1, M):
            if i > 1:
                A[i-1, i-2] = -a[i]
            A[i-1, i-1] = 1 - b[i]
            if i < M-1:
                A[i-1, i] = -c[i]

        for n in reversed(range(N)):
            V[1:M, n] = np.linalg.solve(A, V[1:M, n+1])

    # Crank–Nicolson
    elif method == "crank-nicolson":
        A = np.zeros((M-1, M-1))
        B = np.zeros((M-1, M-1))
        for i in range(1, M):
            if i > 1:
                A[i-1, i-2] = -0.5 * a[i]
                B[i-1, i-2] = 0.5 * a[i]
            A[i-1, i-1] = 1 - 0.5*b[i]
            B[i-1, i-1] = 1 + 0.5*b[i]
            if i < M-1:
                A[i-1, i] = -0.5 * c[i]
                B[i-1, i] = 0.5 * c[i]

        for n in reversed(range(N)):
            b_vec = B @ V[1:M, n+1]
            V[1:M, n] = np.linalg.solve(A, b_vec)

    return np.interp(S0, S, V[:, 0])
