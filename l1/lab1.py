import numpy as np

def is_invertible_and_inverse(A, A_inv, x, i):
    n = A.shape[0]
    
    l = np.dot(A_inv, x)
    
    if l[i] == 0:
        return False, None 
    
    la = l.copy()
    la[i] = -1
    
    lb = (-1 / l[i]) * la
    lb = lb.flatten()  # преобразование (n, 1) в (n,)
    
    Q = np.eye(n)  # единичная матрица
    Q[:, i] = lb  # заменяется i-й столбец на вектор lb
    
    B_inv = np.dot(Q, A_inv)
    
    return True, B_inv


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    A_inv = np.linalg.inv(A)
    
    x = np.array([[7], [8], [9]], dtype=float)
    i = 2
    
    is_invertible, B_inv = is_invertible_and_inverse(A, A_inv, x, i-1)
    
    if is_invertible:
        print("Матрица B обратима.")
        print("Обратная матрица B_inv:")
        print(B_inv)
    else:
        print("Матрица B необратима.")