import numpy as np
import functions as f

def Chen(K, sigma_dl, sigma_n, P, b, tau_min, tau_max, q):
    L = 2
    Nt = L
    Nr = L

    tau_k = np.zeros(K)
    tau_k_Q = np.zeros((K, b))
    tau_k_r = np.zeros(K)
    omitted = np.zeros(K)

    d_k = (np.random.normal(0, sigma_dl, (K, L, 1)) + 1j * np.random.normal(0, sigma_dl, (K, L, 1))) / np.sqrt(2)
    s_k = d_k
    H_k = (np.random.normal(0, sigma_dl, (K, Nr, Nt)) + 1j * np.random.normal(0, sigma_dl, (K, Nr, Nt))) / np.sqrt(2)
    w_k = np.eye(L)
    n = (np.random.normal(0, sigma_n, (L, 1)) + 1j * np.random.normal(0, sigma_n, (L, 1))) / np.sqrt(2)

    # Calculating tau
    for k in range(K):
        H_k_squared = H_k[k] @ np.transpose(H_k[k]).conj()

        # tau_k[k] 계산
        w_s = w_k @ s_k[k]  # 벡터 계산
        tau_k[k] = 1 / np.real(np.transpose(w_s).conj() @ np.linalg.inv(H_k_squared) @ w_s)


    ##### Quantization process #####
    # Quantizing tau
    for k in range(K):
        omitted, tau_k_Q[k] = f.quantize_scalar(tau_k[k], b, tau_min, tau_max)

    # Reconstrcution of tau_Q
    for k in range(K):
        tmp = np.concatenate((omitted, tau_k_Q[k]),0)
        tau_k_r[k] = f.reconsturct_binary(tmp)
        if tau_k_r[k] == 0:
            tau_k_r[k] = tau_min

    eta = 0
    if q == 0:
        eta = P * np.min(tau_k)
    if q == 1:
        eta = P * np.min(tau_k_r)

    A = np.eye(L) / np.sqrt(eta)

    B_k = np.zeros((K, Nt, L), dtype=complex)
    print("q=",q)
    for k in range(K):
        H_k_squared = H_k[k] @ np.transpose(H_k[k]).conj()
        B_k[k] = np.sqrt(eta) * np.transpose(H_k[k]).conj() @ np.linalg.inv(H_k_squared) @ w_k
        Pk =  np.linalg.norm(B_k[k]@s_k[k])
        print(Pk)


    # Calcuating sum of symbols
    d = np.zeros((L, 1))
    d_hat = np.zeros((L, 1))
    for k in range(K):
        d = d + w_k@d_k[k]
        d_hat = d_hat + s_k[k]

    # for k in range(K):
    #     print('##############')
    #     print(d_k[k])
    #     print(A@H_k[k]@B_k[k]@s_k[k])

    d_hat = d_hat + A@n

    e = d-d_hat

    e_norm = np.linalg.norm(e)

    return e_norm

