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
    
    # проверяем на наличие искуственных переменных
    if any(x_opt[n:] != 0):
        return "Задача несовместна"
    
    x = x_opt[:n]  # допустимый план задачи
    B = [j for j in B_opt if j < n]  # Исключаем искусственные переменные

    # Шаг 7-9: Замена искусственных переменных
    while any(j >= n for j in B):
        for k in range(len(B)):
            if B[k] >= n:  # Найдена искусственная переменная
                for j in range(n):
                    if j not in B:
                        l_j = np.linalg.solve(A_tilde[:, B], A_tilde[:, j])
                        if l_j[k] != 0:
                            B[k] = j  # Заменяем искусственную переменную
                            break
                else:
                    # Искусственную переменную заменить не удалось → удаляем строку
                    A_tilde = np.delete(A_tilde, k, axis=0)
                    b = np.delete(b, k)
                    B.pop(k)
                    break
                
    return x, B


c = np.array([1, 0, 0])
A = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([0, 0])
print(initial_simplex_phase(c, A, b))


c2 = np.array([3, 2])
A2 = np.array([[1, 2], [3, 1]])
b2 = np.array([4, 5])
print(initial_simplex_phase(c2, A2, b2))
