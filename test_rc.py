import numpy as np

R = np.logspace(1, 7, 20)
C = np.logspace(-12, -3, 20)

best_err = float('inf')
best_pair = (-1, -1)

for i in range(20):
    for j in range(20):
        rc = R[i] * C[j]
        freq = 1.0 / (1.386 * rc)
        err = abs(freq - 555) / 555
        if err < best_err:
            best_err = err
            best_pair = (i, j)

print("Best R idx:", best_pair[0], "val:", R[best_pair[0]])
print("Best C idx:", best_pair[1], "val:", C[best_pair[1]])
print("Freq:", 1.0 / (1.386 * R[best_pair[0]] * C[best_pair[1]]))
print("Err:", best_err)
