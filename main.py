import numpy as np
import matplotlib.pyplot as plt

import Chen_et_al as Chen

###### Parameters #####
K = 3
sigma_dl = 1  # Squared value
b = 10
P = 10
sigma_n = 1
tau_min = 0.01
tau_max = 0.5
##### Variables #####

b_values = range(1, b+1)  # b from 1 to 10
num_samples = 1  # 100 samples per b

error_by_bits = np.zeros(b)
# for bb in b_values:
#     print('b=',bb)
#     error_samples_bits = []
#     for s in range(num_samples):
#         e_b = Chen.Chen(K, sigma_dl, sigma_n, P, bb, tau_min, tau_max)
#         error_samples_bits.append(e_b)
#     error_average_bits = np.mean(error_samples_bits)
#     error_by_bits[bb-1] = error_average_bits

P_dB = np.arange(0, 31, 5)
P_values = 10 ** (P_dB / 10)  # dB 값을 선형 값으로 변환
error_by_snr = np.zeros(len(P_dB))
error_by_snr_Q = np.zeros(len(P_dB))
for i, snr in enumerate(P_dB):
    print('p=',P_values[i])
    error_samples_snr = []
    error_samples_snr_Q = []
    for s in range(num_samples):
        e_s = Chen.Chen(K, sigma_dl, sigma_n, P_values[i], b, tau_min, tau_max, q=0)
        e_s_Q = Chen.Chen(K, sigma_dl, sigma_n, P_values[i], b, tau_min, tau_max, q=1)
        error_samples_snr.append(e_s)
        error_samples_snr_Q.append(e_s_Q)

    error_average_snr = np.mean(error_samples_snr)
    error_by_snr[i] = error_average_snr
    error_average_snr_Q = np.mean(error_samples_snr_Q)
    error_by_snr_Q[i] = error_average_snr_Q



# plt.figure(figsize=(10, 6))
# plt.plot(b_values, error_by_bits, marker='o', linestyle='-', color='b', label='Average Error')
# plt.title('Average Error vs. Quantization Bits (b)', fontsize=14)
# plt.xlabel('Quantization Bits (b)', fontsize=12)
# plt.ylabel('Average Error', fontsize=12)
# plt.grid(alpha=0.5)
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(P_dB, error_by_snr, marker='o', linestyle='-', color='b', label='Average Error')
plt.plot(P_dB, error_by_snr_Q, marker='^', linestyle='-', color='r', label='Average Error(Q)')
plt.title('Average Error vs. SNR (dB)', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Error', fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.show()