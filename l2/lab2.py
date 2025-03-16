import numpy as np

def is_invertible_and_inverse(A_inv, x, i):
    """
    Реализация нахождения обратной матрицы к заданной при изменении одного столбца.
    """
    n = A_inv.shape[0]
    l = A_inv @ x
    if l[i] == 0:
        return False, None 
    
    la = l.copy()
    la[i] = -1
    lb = (-1 / l[i]) * la
    lb = lb.flatten()
    
    Q = np.eye(n)
    Q[:, i] = lb
    
    B_inv = Q @ A_inv
    return True, B_inv


def simplex_method(c, A, b, B):
    """
    Реализация основной фазы симплекс-метода.
    
    :param c: Вектор коэффициентов целевой функции
    :param A: Матрица ограничений
    :param b: Вектор правых частей ограничений
    :param B: Начальные базисные индексы
    :return: Оптимальный план x или сообщение о неограниченности функции
    """
    m, n = A.shape
    x = np.zeros(n)
    AB = A[:, B]
    A_inv = np.linalg.inv(AB)
    x[B] = np.linalg.solve(AB, b)

    while True:
        AB = A[:, B]
        cB = c[B]
        u = cB @ A_inv  # вектор потенциалов
        delta = u @ A - c  # вектор оценок 
        
        if np.all(delta >= 0):
            return x, B
        
        j0 = np.where(delta < 0)[0][0] 
        z = A_inv @ A[:, j0]  # направление изменений
        
        theta = np.full(m, np.inf)
        valid_indices = z > 0
        theta[valid_indices] = x[B][valid_indices] / z[valid_indices]
        
        if np.all(theta == np.inf):
            return "Целевой функционал задачи не ограничен сверху на множестве допустимых планов."
        
        k = np.argmin(theta)
        j_star = B[k]   # Обновляем базис, убирая из базиса переменную, которая первой достигнет ограничения
        B[k] = j0  

        success, A_inv = is_invertible_and_inverse(A_inv, A[:, j0], k)
        if not success:
            return "Новая базисная матрица необратима."

        x[j0] = theta[k]
        for i in range(m):
            if i != k:
                x[B[i]] -= theta[k] * z[i]
        x[j_star] = 0


if __name__ == "__main__":
    c = np.array([3, 2, 0, 0]) 
    A = np.array([
        [1, 1, 1, 0],
        [2, 1, 0, 1]
    ])
    b = np.array([4, 6])  
    B = np.array([2, 3]) 
    
    result = simplex_method(c, A, b, B)
    print("Оптимальный план:", result)

    # пример из условия
    c = np.array([1, 1, 0, 0, 0])  
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    b = np.array([1, 3, 2]) 
    B = np.array([2, 3, 4]) 
    
    result = simplex_method(c, A, b, B)
    print("Оптимальный план:", result)