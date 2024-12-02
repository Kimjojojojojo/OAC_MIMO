import numpy as np

def quantize_scalar(x, b, min_val, max_val):
    y = 1
    i = 0
    while y > max_val:
        i = i + 1
        y = 2**(-i)
    q_omitted = np.zeros(i-1)

    q_bits = np.zeros(b)
    for j in range(b):
        y = 2**(-i-j)
        tmp = x - y
        if tmp > 0:
            q_bits[j] = 1
            x = tmp
        if tmp < 0:
            q_bits[j] = 0
            x = tmp + y

    #print("Omitted bits: ", q_omitted)
    return q_omitted, q_bits

def reconsturct_binary(x):
    sum = 0
    l = len(x)

    for i in range(l):
        sum = sum + x[i]*2**(-(i+1))

    return sum

def highest_nonzero_index(arr):
    """
    0이 아닌 원소 중 가장 높은 인덱스를 반환.
    없으면 -1 반환.
    """
    for i in range(len(arr) - 1, -1, -1):  # 배열을 역순으로 탐색
        if arr[i] != 0:
            return i
    return -1  # 0이 아닌 원소가 없는 경우