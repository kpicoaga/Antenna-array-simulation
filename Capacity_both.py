import numpy as np
import matplotlib.pyplot as plt
from Z_Heq_Water import compute_Heq_and_Rn, water_filling

# Constants
c = 3e8
a = 0.0025
Ra = 50
grid_size = 7
A = 1
Rr = 10
Rt = 10
Nf = 1.2
T = 300
P_T = 0.2

frequencies_low = np.linspace(3.3e9, 3.7e9, 40)
frequencies_high = np.linspace(17.3e9, 17.7e9, 40)
frequencies = np.concatenate([frequencies_low, frequencies_high])
B_all = [60e3]*40 + [240e3]*40
delta_values = np.linspace(0.003, 0.018, 15)

def compute_capacity(delta_values, mode, D):
    capacity_values = []
    for delta in delta_values:
        lambdas_all = []
        dims = []
        for f in frequencies:
            Heq, Rn = compute_Heq_and_Rn(f, c, a, Ra, delta, D, grid_size, A, Rr, Rt, Nf, T, mode=mode)
            M = Heq.conj().T @ np.linalg.inv(Rn) @ Heq
            lambdas = np.maximum(np.linalg.eigvalsh(M)[::-1], 1e-12)
            lambdas_all.extend(lambdas)
            dims.append(len(lambdas))

        lambdas_all = np.array(lambdas_all)
        power_alloc_all = water_filling(lambdas_all, P_T)
        power_alloc_all *= P_T / np.sum(power_alloc_all)

        capacity = 0
        idx = 0
        for dim, B in zip(dims, B_all):
            p = power_alloc_all[idx:idx + dim]
            lam = lambdas_all[idx:idx + dim]
            capacity += B * np.sum(np.log2(1 + p * lam))
            idx += dim
        capacity_values.append(capacity / 1e6)
    return capacity_values

# Ask user for mode
mode = input("Select mode ('NF' for Near Field, 'FF' for Far Field): ").strip().upper()

if mode == 'NF':
    D = 0.02
elif mode == 'FF':
    D = 900
else:
    print("Invalid mode selected. Please enter 'NF' or 'FF'.")
    exit()

capacity_values = compute_capacity(delta_values, mode, D)

plt.figure(figsize=(9,6))
plt.plot(delta_values * 1e3, capacity_values, marker='o', label=f'{mode} Mode (D={D} m)')
plt.xlabel('Antenna Spacing δ (mm)')
plt.ylabel('Total Multi-band Capacity (Mbps)')
plt.title(f'Capacity vs Antenna Spacing for {mode} Mode')
plt.grid(True)
plt.legend()
plt.show()