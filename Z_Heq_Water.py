import numpy as np
from hex_array import get_hex_positions, get_self_impedance, compute_mutual_impedance

def build_Z_matrix(f, c, positions, Zpp):
    """
    Constructs the mutual impedance matrix Z for a 2D array of antenna elements.

    This matrix includes both self-impedances (on the diagonal) and mutual impedances (off-diagonal)
    based on the distances between antenna positions and a fixed broadside orientation (theta = π/2).

    Parameters:
        f : Frequency of operation in Hz.
        c : Speed of radio frequency radiation (in m/s)
        positions : Array of shape (N, 2) with (x, y) positions of the N antenna elements.
        Zpp (complex): Self-impedance of each antenna element (TM₁ mode).

    Returns:
      Complex mutual impedance matrix Z of shape (N, N), where Z[i, j] represents the 
      impedance between elements i and j.
    """
    beta= np.pi / 2  # Broadside orientation
    N = len(positions)
    Zpp_real = Zpp.real
    Z = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Z[i, j] = Zpp
            else:   
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                beta_ij = np.pi/2 + np.arctan2(dy, dx) - np.pi / 2 
                Z[i, j] = compute_mutual_impedance(f, c, Zpp_real, Zpp_real, dist, beta_ij)
    return Z

def build_Z_RT_matrix(f, c, positions, Zpp, D):
    """
    Constructs the mutual impedance matrix Z_RT between a single receiver and an array of N transmit antennas.

    The receiver is positioned directly above the center of the array at height D,
    and each mutual impedance is computed based on the 3D distance from the receiver to each transmit element.

    Parameters:
        f : Frequency of operation in Hz.
        c : Speed of radio frequency propagation in m/s.
        positions: Array of shape (N, 2) with (x, y) positions of the N transmit antennas.
        Zpp : Self-impedance of the TM₁ mode for each transmit antenna.
        D: Vertical distance (in meters) from the antenna plane to the receiver.

    Returns:
        Row vector Z_RT of shape (1, N), where Z_RT[0, j] is the mutual impedance
        between the receiver and the j-th transmit antenna.
    """
    N = len(positions) # Number of transmit antennas
    Zpp_real = Zpp.real
    Z_RT = np.zeros((1, N), dtype=complex)
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1]) 
    receiver_pos = np.array([center_x, center_y, D]) # Receiver position above the center of the array

    for j in range(N):
    # Relative position of Tx antenna with respect to array center
        dx = positions[j][0] - center_x
        dy = positions[j][1] - center_y

        # distance
        r = np.sqrt(dx**2 + dy**2)
        # Distance to Rx antenna assuming broadside arrays 
        dist = np.sqrt(r**2 + D**2)
        # Gamma: angle having tilt theta = pi/2
        gamma_q = np.arctan2(dy, dx)
        # cos(beta) having tilt theta = pi/2
        cos_beta = (r ) / dist
        # Beta angle
        beta_j = np.arccos(cos_beta)

        Z_RT[0, j] = compute_mutual_impedance(f, c, Zpp_real, Zpp_real, dist, beta_j)

    return Z_RT

def compute_Heq_and_Rn(f, c, a, Ra, delta, D, grid_size, A, Rr, Rt, Nf, T):
    """
    Computes the equivalent channel matrix (H_eq) and the noise covariance matrix (Rn) 
    for a tightly coupled hexagonal antenna array operating in the near field.

    Parameters:
        f         : Operating frequency (Hz)
        c         : Speed of electromagnetic wave propagation (m/s)
        a         : Radius of the enclosing sphere for each antenna element (m)
        Ra        : Equivalent resistance of the antenna in the TM1 mode (Ohms)
        delta     : Spacing between adjacent antenna elements (m)
        D         : Distance from the transmitter array to the receiver (m)
        grid_size : Number of elements along one side of the hexagonal array (results in N = grid_size² elements)
        A         : Frequency-independent amplifier gain of the LNA (unitless)
        Rr        : Input resistance of the receiver’s low-noise amplifier (Ohms)
        Rt        : Input resistance of each transmit chain (Ohms)
        Nf        : Noise figure of the receiver (unitless)
        T         : System temperature (Kelvin)

    Returns:
        H_eq :(1×N)
            Equivalent channel matrix including antenna coupling and impedance effects.

        Rn   :(1×1)
            Noise covariance matrix at the receiver output, accounting for LNA noise and thermal noise 
            influenced by mutual coupling and impedance matching.
    """
    
    positions = get_hex_positions(grid_size, delta)
    N = len(positions)
    Zpp = get_self_impedance(f, c, a, Ra)
    Z_R = np.array([[Zpp]])
    Z_T = build_Z_matrix(f, c, positions, Zpp)

    Z_RT = build_Z_RT_matrix(f, c, positions, Zpp, D)
    Z_TR = Z_RT.conj().T
    I_rx = np.eye(1)
    I_tx = np.eye(N)

    U = np.linalg.inv(Z_R + Rr * I_rx)
    V = np.linalg.inv(Z_T + Rt * I_tx - Z_TR @ U @ Z_RT)

    W = A * Rr * U @ (Z_RT @ V @ Z_TR @ U + I_rx)
    H_eq = A * Rr * U @ Z_RT @ V
    
    kB = 1.38e-23
    Rn = 4 * kB * T * (W @ np.real(Z_R) @ W.conj().T + Rr * (Nf - 1) * I_rx)

    return H_eq, Rn

def water_filling(lambdas, P_T):
    """
    Performs the water-filling power allocation algorithm.

    Parameters:
        lambdas : Channel eigenvalues sorted in any order.
        P_T : Total available transmit power.

    Returns:
        power_alloc_unsorted : Power allocated to each channel eigenmode (same order as input lambdas).
    """
    n = len(lambdas)
    sorted_idx = np.argsort(lambdas)[::-1]
    lambdas_sorted = lambdas[sorted_idx]
    power_alloc = np.zeros(n)
    for k in range(n, 0, -1):
        mu = (P_T + np.sum(1 / lambdas_sorted[:k])) / k
        if k == n or mu > 1 / lambdas_sorted[k - 1]:
            power_alloc[:k] = mu - 1 / lambdas_sorted[:k]
            power_alloc[power_alloc < 0] = 0
            break
    power_alloc_unsorted = np.zeros(n)
    power_alloc_unsorted[sorted_idx] = power_alloc
    return power_alloc_unsorted