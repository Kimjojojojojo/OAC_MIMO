import numpy as np
import matplotlib.pyplot as plt

# Zhu 함수 정의
def Zhu(K, at, L, sigma_dl, sigma_n, P_0):
    Nt = 1
    Nr = at

    H_k = (np.random.normal(0, 1, (K, Nr, Nt)) + 1j * np.random.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)

    U_k = np.zeros((K, Nr, Nr), dtype=complex)
    lambda_min_k = np.zeros(K)

    # Eigenvalue & eigenvector extraction
    for k in range(K):
        eigenvectors, eigenvalues, _ = np.linalg.svd(H_k[k])
        # print(eigenvalues)
        # print(eigenvectors)
        U_k[k] = eigenvectors
        lambda_min_k[k] = eigenvalues[0]**2

    G = np.zeros((Nr, Nr), dtype=complex)
    for k in range(K):
        G = G + lambda_min_k[k] * U_k[k] @ np.transpose(U_k[k]).conj()

    V_G, Sigma_G, V_GH = np.linalg.svd(G, full_matrices=False)
    F_star = V_G[:, :L]

    # True optimal eta
    eta_k = np.zeros(K)
    for k in range(K):
        eta_k[k] = np.real(
            np.linalg.trace(np.transpose(F_star).conj() @ H_k[k] @ np.transpose(H_k[k]).conj() @ F_star)) ** (-1)

    eta_star = np.max(eta_k)
    return eta_star

# Parameters
K = 5
L = 2
sigma_dl = 1
sigma_n = 0.1
P_0 = 1
at = 10  # 고정된 at 값
num_samples = 1000  # 반복 수

# Numerical calculation
eta_samples = []

for s in range(num_samples):
    print(f'{s + 1}/{num_samples} samples')
    eta_k = Zhu(K, at, L, sigma_dl, sigma_n, P_0)
    eta_samples.append(eta_k)

eta_samples = np.array(eta_samples)  # 리스트를 NumPy 배열로 변환
print(eta_samples)
print(sum(eta_samples))
# Plot PDF
plt.figure(figsize=(8, 6))
plt.hist(eta_samples, bins=50, density=True, alpha=0.7, color='blue', label=f"at = {at}")
plt.xlabel(r"$\eta_k$")
plt.xlim([0,30])
plt.ylabel("PDF")
plt.title(f"PDF of $\eta_k$ for at = {at}")
plt.legend(loc='upper right')
plt.grid()
plt.show()
