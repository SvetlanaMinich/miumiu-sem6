import numpy as np
from lab2 import simplex_method

def initial_simplex_phase(c, A, b):
    m, n = A.shape
    
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
    
    c_tilde = np.concatenate((np.zeros(n), -np.ones(m)))  # целевая функция
    A_tilde = np.hstack((A, np.eye(m))) 

    x_tilde = np.concatenate((np.zeros(n), b)) 
    B = list(range(n, n + m)) 
    
    x_opt, B_opt = simplex_method(c_tilde, A_tilde, b, B)
    # print(x_opt, B_opt)
    
    if any(x_opt[n:] != 0):
        return "Задача несовместна"
    
    x = x_opt[:n]  # допустимый план задачи

    # Шаг 7-9: Замена искусственных переменных
    while any(j >= n for j in B_opt):
        for k in range(len(B_opt)):
            if B[k] >= n:  # Найдена искусственная переменная
                for j in range(n):
                    if j not in B_opt:
                        l_j = np.linalg.solve(A_tilde[:, B_opt], A_tilde[:, j])
                        if l_j[k] != 0:
                            B_opt[k] = j 
                            break
                    else:
                        # Искусственную переменную заменить не удалось → удаляем строку
                        A = np.delete(A, k, axis=0)
                        b = np.delete(b, k)
                        B_opt.pop(k)
                        break
                
    return x, B_opt, A, b


c = np.array([1, 0, 0])
A = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([0, 0])
print(initial_simplex_phase(c, A, b))
