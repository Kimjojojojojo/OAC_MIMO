import numpy as np
import functions as f

Nt = 4
eta_min = [0.1]
eta_max = [1]
M = 10 # feedback round
K = 3
Q = np.linspace(eta_min[0], eta_max[0], Nt)
m_k = np.zeros(K)
eta_k = 0.5
print(Q)
for k in range(K):
    m_k[k] = np.argmin(np.abs(eta_k - Q))
print(m_k)

print(eta_k-Q)

print(np.argmin(Q))