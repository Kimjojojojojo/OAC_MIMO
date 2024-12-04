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
        idx = np.argsort(eigenvalues)[::-1]  # 내림차순 정렬 (내림차순이 필요 없으면 [::1])
        sorted_eigenvalues = np.real(eigenvalues[idx])
        sorted_eigenvectors = eigenvectors[:, idx]

        U_k[k] = sorted_eigenvectors
        lambda_min_k[k] = sorted_eigenvalues[-1] ** 2

    G = np.zeros((Nr, Nr), dtype=complex)
    for k in range(K):
        G = G + lambda_min_k[k] * U_k[k] @ np.transpose(U_k[k]).conj()

    V_G, Sigma_G, V_GH = np.linalg.svd(G, full_matrices=False)

    F_star = V_G[:, 0:L]

    ##### True optimal eta #####
    eta_k = np.zeros(K)
    for k in range(K):
        eta_k[k] = (1 / P_0) * np.real(
            np.linalg.trace(np.transpose(F_star).conj() @ H_k[k] @ np.transpose(H_k[k]).conj() @ F_star)) ** (-1)

    eta_star = np.max(eta_k)
    return eta_star

# Parameters
K = 30
L = 10
sigma_dl = 1
sigma_n = 0.1
P_0 = 10
at_values = [10, 20, 30, 40, 50]  # 다양한 at 값
num_samples = 1000  # 반복 수

# CDF 계산 및 플롯
plt.figure(figsize=(8, 6))

for at in at_values:
    # Numerical calculation
    eta_samples = []

    for s in range(num_samples):
        print(f'at = {at}, {s + 1}/{num_samples} samples')
        eta_k = Zhu(K, at, L, sigma_dl, sigma_n, P_0)
        eta_samples.append(eta_k)

    eta_samples = np.array(eta_samples)  # 리스트를 NumPy 배열로 변환

    # Calculate CDF
    sorted_eta = np.sort(eta_samples)  # 오름차순 정렬
    cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)  # 누적 비율 계산

    # Plot CDF
    plt.plot(sorted_eta, cdf, marker='.', linestyle='-', label=f"Nr = {at}")

# 그래프 설정
plt.xlabel(r"$\eta_k$")
plt.xlim([0, 0.1])
plt.ylabel("CDF")
plt.title(f"CDF of $\eta_k$ for various at values, L={L}, K={K}")
plt.legend(loc='lower right')
plt.grid()
plt.show()
