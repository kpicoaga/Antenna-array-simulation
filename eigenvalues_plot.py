import numpy as np
import matplotlib.pyplot as plt
from Z_Heq_Water import compute_Heq_and_Rn

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
delta = 0.006  # fixed value for this plot

# Frequency range: 2 GHz to 20 GHz
frequencies = np.linspace(2e9, 20e9, 200)
max_eigenvalue_db = []

for f in frequencies:
    Heq, Rn = compute_Heq_and_Rn(f, c, a, Ra, delta, D, grid_size, A, Rr, Rt, Nf, T)
    M = Heq.conj().T @ np.linalg.inv(Rn) @ Heq
    lambdas = np.maximum(np.linalg.eigvalsh(M), 1e-12)  
    max_lambda_db = 10 * np.log10(np.max(lambdas)) # Convert to dB
    max_eigenvalue_db.append(max_lambda_db) 

# Plot only the max eigenvalue (dominant mode)
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1e9, max_eigenvalue_db, label="Dominant SNR Mode", color="navy", linewidth=2)

plt.xlabel("Frequency (GHz)")
plt.ylabel(" λ (dB)")
plt.title(r"λ vs Frequency (δ = 6 mm, $\theta = \pi/2$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()