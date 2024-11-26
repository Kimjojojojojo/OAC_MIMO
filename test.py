import numpy as np
import functions as f

x = 0.05
b = 5
min_val = 0.01
max_val = 0.1

O, Q = f.quantize_scalar(x, b, min_val, max_val)
concat = np.concatenate((O, Q), axis=0)
print(O)
print(Q)
print(concat)