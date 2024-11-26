import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 10
L = K
Nt = L
Nr = L
sigma_dl = 1  # Squared value
num_samples = 10000  # Number of samples for tau_k

# Storage for tau_k samples
tau_k_samples = []

for _ in range(num_samples):
    # Variables for each sample
    d_k = (np.random.normal(0, sigma_dl, (K, L, 1)) + 1j * np.random.normal(0, sigma_dl, (K, L, 1))) / np.sqrt(2)
    s_k = d_k
    H_k = (np.random.normal(0, sigma_dl, (K, Nr, Nt)) + 1j * np.random.normal(0, sigma_dl, (K, Nr, Nt))) / np.sqrt(2)
    w_k = np.eye(L)

    tau_k = np.zeros(K)

    for k in range(K):
        # Compute B_k[k]
        H_k_squared = H_k[k] @ np.transpose(H_k[k]).conj()
        w_s = w_k @ s_k[k]  # Vector computation

        # Compute tau_k[k]
        tau_k[k] = 1 / np.real(np.transpose(w_s).conj() @ np.linalg.inv(H_k_squared) @ w_s)

    # Store the average tau_k across K users for this sample
    tau_k_samples.append(np.mean(tau_k))

# Convert samples to NumPy array for analysis
tau_k_samples = np.array(tau_k_samples)

# Plot PDF of tau_k
plt.figure(figsize=(10, 6))
plt.hist(tau_k_samples, bins=50, density=True, alpha=0.7, label='Empirical PDF')
plt.title(r'PDF of $\tau_k$', fontsize=14)
plt.xlabel(r'$\tau_k$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(alpha=0.4)
plt.legend()
plt.show()
