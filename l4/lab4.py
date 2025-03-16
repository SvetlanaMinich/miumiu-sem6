import numpy as np

def dual_simplex(c, A, b, B, max_iter=100):
    """
    Функция dual_simplex решает задачу линейного программирования методом двойственного симплекса.
    
    Параметры:
    - c: вектор коэффициентов целевой функции (numpy массив формы (n,))
    - A: матрица ограничений (numpy массив формы (m, n))
    - b: вектор правых частей ограничений (numpy массив формы (m,))
    - B: список базисных индексов (индексация с 0, например, [3, 4] для j4 и j5 из текста)
    - max_iter: максимальное число итераций, чтобы избежать бесконечного цикла

    Возвращает:
    - оптимальный псевдоплан (вектор x), если найден оптимум,
    - сообщение о том, что задача не совместна, если такой случай выявлен.
    """
    n = len(c)     
    m = A.shape[0]  

    for iter_count in range(max_iter):
        A_B = A[:, B]
        try:
            # инвертируем матрицу
            A_B_inv = np.linalg.inv(A_B)
        except np.linalg.LinAlgError:
            return "Метод не может быть применён."
        
        c_B = c[B]
        
        # Базисный допустимый план двойственной задачи
        y = A_B_inv.T @ c_B  
        
        # Псевдоплан
        k_B = A_B_inv @ b
        k = np.zeros(n)
        for idx, basis_index in enumerate(B):
            k[basis_index] = k_B[idx]
        
        if np.all(k >= 0):
            return k
        
        # Находим первую отрицательную базисную переменную псевдоплана
        for idx, basis_index in enumerate(B):
            if k[basis_index] < 0:
                row_index = idx
                break
        
        delta_y = A_B_inv[row_index, :]
        
        # находим 
        mu = {}
        non_basis = [j for j in range(n) if j not in B]
        for j in non_basis:
            mu[j] = np.dot(delta_y, A[:, j])
        
        if all(mu[j] >= 0 for j in non_basis):
            return "Задача не совместна (не существует допустимого плана)."
        
        # вычисляем наименьшее отклонение переменной (чтобы изменения были минимальными)
        sigma = {}
        for j in non_basis:
            if mu[j] < 0:
                sigma[j] = (c[j] - np.dot(A[:, j], y)) / mu[j]
        
        j0 = min(sigma, key=sigma.get)
        
        B[row_index] = j0

        print(f"Итерация {iter_count}")
        
    return "Превышено максимальное число итераций."


c = np.array([-4, -3, -7, 0, 0], dtype=float)
A = np.array([[-2, -1, -4, 1, 0],
              [-2, -2, -2, 0, 1]], dtype=float)
b = np.array([-1, -1.5], dtype=float)
B_initial = [3, 4]

print(dual_simplex(c, A, b, B_initial))
