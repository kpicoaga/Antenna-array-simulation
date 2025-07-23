# hex_array.py
import numpy as np

def get_hex_positions(grid_size, delta):
    positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * delta + (row % 2) * (delta / 2)
            y = row * (np.sqrt(3) / 2) * delta
            positions.append(np.array([x, y]))
    return np.array(positions)

def get_self_impedance(f, c, a, Ra):
    omega = 2 * np.pi * f
    num = c**2 + 1j * omega * c * a - (omega * a)**2
    denom = 1j * omega * c * a - (omega * a)**2
    return (num / denom) * Ra

def compute_mutual_impedance(f, c, Zpp_real_p, Zpp_real_q, delta_pq, beta):
    k0 = 2 * np.pi * f / c
    jk0delta = 1j * k0 * delta_pq
    term1 = (0.5 * np.sin(beta)**2) * (1 / jk0delta + 1 / jk0delta**2 + 1 / jk0delta**3)
    term2 = (np.cos(beta)**2) * (1 / jk0delta**2 + 1 / jk0delta**3)
    return -3 * np.sqrt(Zpp_real_p) * np.sqrt(Zpp_real_q) * (term1 - term2) * np.exp(-jk0delta)