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