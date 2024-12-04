import numpy as np

import functions as f
##### Parameters

###### Variables #####


def Zhu(K, at, L, sigma_dl, sigma_n, P_0):
    Nt = 1
    Nr = at

    H_k = (np.random.normal(0, 1, (K, Nr, Nt)) + 1j * np.random.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)

    U_k = np.zeros((K, Nr, Nr), dtype=complex)
    lambda_min_k = np.zeros(K)

    # eigenvlue & vector extraction
    for k in range(K):
        eigenvectors, eigenvalues, _ = np.linalg.svd(H_k[k])
        idx = np.argsort(eigenvalues)[::-1]  # 내림차순 정렬 (내림차순이 필요 없으면 [::1])
        sorted_eigenvalues = np.real(eigenvalues[idx])
        sorted_eigenvectors = eigenvectors[:, idx]

        U_k[k] = sorted_eigenvectors
        lambda_min_k[k] = sorted_eigenvalues[-1]**2

    G = np.zeros((Nr, Nr), dtype=complex)
    for k in range(K):
        G = G + lambda_min_k[k] * U_k[k]@np.transpose(U_k[k]).conj()

    V_G, Sigma_G, V_GH = np.linalg.svd(G, full_matrices=False)


    F_star = V_G[:, 0:L]

    ##### True optimal eta #####
    eta_k = np.zeros(K)
    for k in range(K):
        eta_k[k] = (1/P_0) * np.real(np.linalg.trace(np.transpose(F_star).conj()@H_k[k]@np.transpose(H_k[k]).conj()@F_star))**(-1)

    eta_star = np.max(eta_k)
    ############################

    ###### Quantizing eta #####
    eta_min = 0.1 * (1/P_0)
    eta_max = 1 * (1/P_0)
    M = 10 # feedback round
    eta_star_hat = 1
    for m in range(M):
        # 1)
        Delta = (eta_max - eta_min)/L
        # 2)
        Q = np.linspace(eta_min, eta_max, L)
        # 3)
        e_k = np.zeros((K, L))


        for k in range(K):
            m_k = np.argmin(np.abs(eta_k[k] - Q))
            e_k[k][m_k] = 1
        e_sum = np.sum(e_k,0)
        l_max = f.highest_nonzero_index(e_sum)

        # 4)
        eta_star_hat = eta_k[l_max]
        eta_min = eta_star_hat - Delta/2
        eta_max = eta_star_hat + Delta/2

    eta_star_hat = eta_star_hat
    #print(eta_star, (eta_star-eta_star_hat)/eta_star * 100)
    ##########################
    A_star = np.sqrt(eta_star_hat)*F_star

    B_k_star = np.zeros((K, Nt, L), dtype=complex)
    for k in range(K):
        tmp = np.transpose(A_star).conj()@H_k[k]
        if Nt == 1:
            B_k_star[k] = np.transpose(tmp).conj() / (np.linalg.norm(tmp))
        else:
            B_k_star[k] = np.transpose(tmp).conj() @ np.linalg.inv(tmp@np.transpose(tmp).conj())

    s_k = (np.random.normal(0, sigma_dl, (K, L, 1)) + 1j * np.random.normal(0, sigma_dl, (K, L, 1))) / np.sqrt(2)
    n = (np.random.normal(0, sigma_n, (Nr, 1)) + 1j * np.random.normal(0, sigma_n, (Nr, 1))) / np.sqrt(2)

    s = np.zeros((L, 1))
    s_hat = np.zeros((L, 1))
    for k in range(K):
        s = s + s_k[k]
        s_hat = s_hat + np.transpose(A_star).conj()@H_k[k]@B_k_star[k]@s_k[k]
    s_hat = s_hat + np.transpose(A_star).conj()@n

    e = s - s_hat

    e_norm = np.linalg.norm(e)

    if eta_star_hat >= 1:
        print('###########')
        print(eta_star_hat)
        print('###########')
    return abs(eta_star - eta_star_hat)
