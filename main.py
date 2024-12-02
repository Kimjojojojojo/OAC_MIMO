import numpy as np
import matplotlib.pyplot as plt

import Chen_et_al as Chen
import Zhu_et_al as Zhu
###### Parameters #####
K = 3
L = 2
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
#         e_b = Chen.Chen(K, L, sigma_dl, sigma_n, P, bb, tau_min, tau_max)
#         error_samples_bits.append(e_b)
#     error_average_bits = np.mean(error_samples_bits)
#     error_by_bits[bb-1] = error_average_bits

P_dB = np.arange(0, 31, 5)
P_values = 10 ** (P_dB / 10)  # dB 값을 선형 값으로 변환
error_by_snr_Chen = np.zeros(len(P_dB))
error_by_snr_Q_Chen = np.zeros(len(P_dB))
for i, snr in enumerate(P_dB):

    error_samples_snr_Chen = []
    error_samples_snr_Q_Chen = []
    for s in range(num_samples):
        e_s_Chen = Chen.Chen(K, L, sigma_dl, sigma_n, P_values[i], b, tau_min, tau_max, q=0)
        e_s_Q_Chen = Chen.Chen(K, L, sigma_dl, sigma_n, P_values[i], b, tau_min, tau_max, q=1)
        error_samples_snr_Chen.append(e_s_Chen)
        error_samples_snr_Q_Chen.append(e_s_Q_Chen)

    error_average_snr_Chen = np.mean(error_samples_snr_Chen)
    error_by_snr_Chen[i] = error_average_snr_Chen
    error_average_snr_Q_Chen = np.mean(error_samples_snr_Q_Chen)
    error_by_snr_Q_Chen[i] = error_average_snr_Q_Chen

error_by_snr_Zhu = np.zeros(len(P_dB))
for i, snr in enumerate(P_dB):
    print(f'##############{snr}#################')
    error_samples_snr_Zhu = []
    for s in range(num_samples):
        e_s_Zhu = Zhu.Zhu(K, L, sigma_dl, sigma_n, P_values[i])
        error_samples_snr_Zhu.append(e_s_Zhu)

    error_average_snr_Zhu = np.mean(error_samples_snr_Zhu)
    error_by_snr_Zhu[i] = error_average_snr_Zhu


# plt.figure(figsize=(10, 6))
# plt.plot(b_values, error_by_bits, marker='o', linestyle='-', color='b', label='Average Error')
# plt.title('Average Error vs. Quantization Bits (b)', fontsize=14)
# plt.xlabel('Quantization Bits (b)', fontsize=12)
# plt.ylabel('Average Error', fontsize=12)
# plt.grid(alpha=0.5)
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(P_dB, error_by_snr_Chen, marker='o', linestyle='-', color='b', label='Chen')
plt.plot(P_dB, error_by_snr_Q_Chen, marker='^', linestyle='-', color='r', label='Chen(Q)')
plt.plot(P_dB, error_by_snr_Zhu, marker='v', linestyle='-', color='g', label='Zhu')
plt.title('Chen : Average Error vs. SNR (dB)', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Error', fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(P_dB, error_by_snr_Zhu, marker='o', linestyle='-', color='b', label='Average Error')
# plt.title('Zhu : Average Error vs. SNR (dB)', fontsize=14)
# plt.xlabel('SNR (dB)', fontsize=12)
# plt.ylabel('Average Error', fontsize=12)
# plt.grid(alpha=0.5)
# plt.legend()
# plt.show()