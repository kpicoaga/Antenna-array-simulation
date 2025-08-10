import numpy as np
import matplotlib.pyplot as plt
from Z_Heq_Water import compute_Heq_and_Rn, water_filling

# Constants
c = 3e8
a = 0.004
Ra = 50
grid_size = 7
A = 10
Rr = 10
Rt = 10
Nf = 1.2
T = 300
D = 0.2
P_T = 20

# Frequency bands (
frequencies_low = np.linspace(3.3e9, 3.7e9, 40)
frequencies_high = np.linspace(17.3e9, 17.7e9, 40)
frequencies = np.concatenate([frequencies_low, frequencies_high])
B_all = [60e3]*40 + [240e3]*40

# Delta from 0.006 to 0.018 meters
delta_values = np.linspace(0.003, 0.018, 17)
capacity_values = []

for delta in delta_values:
    lambdas_all = []
    dims = []
    for f in frequencies:
        Heq, Rn = compute_Heq_and_Rn(f, c, a, Ra, delta, D, grid_size, A, Rr, Rt, Nf, T)
        M = Heq.conj().T @ np.linalg.inv(Rn) @ Heq 
        lambdas = np.maximum(np.linalg.eigvalsh(M)[::-1], 1e-12)# # Ensure positive eigenvalues
        lambdas_all.extend(lambdas)  
        dims.append(len(lambdas)) 

    lambdas_all = np.array(lambdas_all) 
    power_alloc_all = water_filling(lambdas_all, P_T) 
    power_alloc_all *= P_T / np.sum(power_alloc_all) # Normalize total transmit power

    capacity = 0
    idx = 0 
    for dim, B in zip(dims, B_all): # Loop through each frequency band
        p = power_alloc_all[idx:idx+dim]   
        lam = lambdas_all[idx:idx+dim]  
        capacity += B * np.sum(np.log2(1 + p * lam))  
        idx += dim 
    capacity_values.append(capacity / 1e6)  # in Mbps

# Plot
plt.figure(figsize=(8,5))
plt.plot(delta_values * 1e3, capacity_values, marker='o', label=r'Hexagonal Array, $\theta = \pi/2$')
plt.xlabel('Antenna Spacing δ (mm)')
plt.ylabel('Total Multi-band Capacity (Mbps)')
plt.grid(True)
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
plt.show()
