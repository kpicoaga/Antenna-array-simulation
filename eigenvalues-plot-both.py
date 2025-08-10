import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Z_Heq_Water import compute_Heq_and_Rn

# Constants
c = 3e8                                                     
a = 0.0025
Ra = 50
grid_size = 7
A = 1
Rr = 10
Rt = 10
Nf = 1.2
T = 3001
delta = 0.006  # fixed value for this plot

# Frequency range: 2 GHz to 20 GHz
frequencies = np.linspace(2e9, 20e9, 200)
max_eigenvalue_linear = []


# Ask user for mode selection
mode = input("Select mode ('NF' for Near Field, 'FF' for Far Field): ").strip().upper()

if mode == 'NF':
    D = 0.02
elif mode == 'FF':
    D = 900
else:
    print("Invalid mode selected. Please enter 'NF' or 'FF'.")
    exit()

for f in frequencies:
    Heq, Rn = compute_Heq_and_Rn(f, c, a, Ra, delta, D, grid_size, A, Rr, Rt, Nf, T, mode)
    M = Heq.conj().T @ np.linalg.inv(Rn) @ Heq
    lambdas = np.maximum(np.linalg.eigvalsh(M), 1e-12)
    max_lambda = np.max(lambdas)

    max_eigenvalue_linear.append(max_lambda)



# Plot raw (optional) and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1e9, 10 * np.log10(max_eigenvalue_linear), label="Dominant SNR Mode", color="navy", linewidth=2)
plt.xlabel("Frequency (GHz)")
plt.ylabel("λ (dB)")
plt.title(r"λ vs Frequency (δ = 6 mm, $\theta = \pi/2$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()