import numpy as np
import matplotlib.pyplot as plt

# Zhu 함수 정의
def Zhu(K, at, L, sigma_dl, sigma_n, P_0):
    Nt = 1
    Nr = at

    H_k = (np.random.normal(0, 1, (Nr, Nt)) + 1j * np.random.normal(0, 1, (Nr, Nt))) / np.sqrt(2)

    # Eigenvalue & eigenvector extraction
    eigenvectors, eigenvalues, _ = np.linalg.svd(H_k)

    U_k = eigenvectors
    lambda_min_k = eigenvalues

    G = lambda_min_k * U_k @ np.transpose(U_k).conj()

    V_G, Sigma_G, V_GH = np.linalg.svd(G, full_matrices=False)
    F_star = V_G[:, :L]

    # True optimal eta
    eta_k = (1 / P_0) * np.real(
        np.linalg.trace(np.transpose(F_star).conj() @ H_k @ np.transpose(H_k).conj() @ F_star)
    ) ** (-1)
    return eta_k

# Parameters
K = 50
L = 10
sigma_dl = 1
sigma_n = 0.1
P_0 = 10
at = 100  # 고정된 at 값
num_samples = 1000  # 반복 수

# Numerical calculation
eta_samples = []

for s in range(num_samples):
    print(f'{s + 1}/{num_samples} samples')
    eta_k = Zhu(K, at, L, sigma_dl, sigma_n, P_0)
    eta_samples.append(eta_k)

eta_samples = np.array(eta_samples)  # 리스트를 NumPy 배열로 변환

# Calculate CDF
sorted_eta = np.sort(eta_samples)  # 오름차순 정렬
cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)  # 누적 비율 계산

# Plot CDF
plt.figure(figsize=(8, 6))
plt.plot(sorted_eta, cdf, marker='.', linestyle='-', color='blue', label=f"at = {at}")
plt.xlabel(r"$\eta_k$")
plt.ylabel("CDF")
plt.title(f"CDF of $\eta_k$ for at = {at}")
plt.legend(loc='lower right')
plt.grid()
plt.show()
