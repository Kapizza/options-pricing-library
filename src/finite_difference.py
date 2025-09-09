import numpy as np

def finite_difference(S0, K, T, r, sigma, Smax=200, M=200, N=2000,
                      method="explicit", option="call"):
    dt = T / N
    dS = Smax / M
    S = np.linspace(0, Smax, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Payoff at maturity (t = T)
    if option == "call":
        V[:, -1] = np.maximum(S - K, 0.0)
    else:
        V[:, -1] = np.maximum(K - S, 0.0)

    # Boundary conditions for 0 <= t <= T
    t_index = np.arange(N + 1)
    if option == "call":
        V[0, :]  = 0.0
        V[-1, :] = Smax - K * np.exp(-r * dt * (N - t_index))
    else:
        V[0, :]  = K * np.exp(-r * dt * (N - t_index))
        V[-1, :] = 0.0

    # Coefficients on the asset grid i = 0..M
    i = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * i**2 - r * i)          # lower diag (i-1)
    b = -dt * (sigma**2 * i**2 + r)                   # center
    c = 0.5 * dt * (sigma**2 * i**2 + r * i)          # upper diag (i+1)

    if method == "explicit":
        # (stability tip) choose dt small enough, e.g. dt <= (dS**2)/(sigma**2*Smax**2)
        for n in range(N-1, -1, -1):
            # interior nodes
            V[1:M, n] = (
                a[1:M] * V[0:M-1, n+1] +
                (1 + b[1:M]) * V[1:M, n+1] +
                c[1:M] * V[2:M+1, n+1]
            )
            # enforce boundaries (already set, but safe to keep)
            V[0, n]  = V[0, n]
            V[M, n]  = V[M, n]

    elif method == "implicit":
        # A * V^n_int = RHS, with boundary contributions in RHS
        A = np.zeros((M-1, M-1))
        for k in range(1, M):
            if k > 1:
                A[k-1, k-2] = -a[k]
            A[k-1, k-1] = 1 - b[k]
            if k < M-1:
                A[k-1, k] = -c[k]

        for n in range(N-1, -1, -1):
            rhs = V[1:M, n+1].copy()
            # add boundaries at time n (Dirichlet, known)
            rhs[0]  += a[1]     * V[0, n]
            rhs[-1] += c[M-1]   * V[M, n]
            V[1:M, n] = np.linalg.solve(A, rhs)

    elif method == "crank-nicolson":
        # (I - 1/2 L) V^n = (I + 1/2 L) V^{n+1} + boundary terms
        A = np.zeros((M-1, M-1))
        B = np.zeros((M-1, M-1))
        for k in range(1, M):
            if k > 1:
                A[k-1, k-2] = -0.5 * a[k]
                B[k-1, k-2] =  0.5 * a[k]
            A[k-1, k-1] = 1 - 0.5 * b[k]
            B[k-1, k-1] = 1 + 0.5 * b[k]
            if k < M-1:
                A[k-1, k] = -0.5 * c[k]
                B[k-1, k] =  0.5 * c[k]

        for n in range(N-1, -1, -1):
            rhs = B @ V[1:M, n+1]
            # boundary contributions at times n and n+1
            rhs[0]  += 0.5 * a[1]   * (V[0, n+1] + V[0, n])
            rhs[-1] += 0.5 * c[M-1] * (V[M, n+1] + V[M, n])
            V[1:M, n] = np.linalg.solve(A, rhs)

    else:
        raise ValueError("method must be 'explicit', 'implicit', or 'crank-nicolson'")

    # Return curve or point(s)
    if np.isscalar(S0):
        return np.interp(S0, S, V[:, 0])
    else:
        return np.interp(np.asarray(S0), S, V[:, 0])
