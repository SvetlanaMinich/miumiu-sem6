import numpy as np

def dual_simplex(c, A, b, B, tol=1e-8, max_iter=100):
    """
    Функция dual_simplex решает задачу линейного программирования методом двойственного симплекса.
    
    Параметры:
    - c: вектор коэффициентов целевой функции (numpy массив формы (n,))
    - A: матрица ограничений (numpy массив формы (m, n))
    - b: вектор правых частей ограничений (numpy массив формы (m,))
    - B: список базисных индексов (индексация с 0, например, [3, 4] для j4 и j5 из текста)
    - tol: допустимая погрешность для проверки неотрицательности (по умолчанию 1e-8)
    - max_iter: максимальное число итераций, чтобы избежать бесконечного цикла

    Возвращает:
    - оптимальный псевдоплан (вектор x), если найден оптимум,
    - сообщение о том, что задача не совместна, если такой случай выявлен.
    """
    n = len(c)     
    m = A.shape[0]  
    
    iter_count = 0

    for iter_count in range(max_iter):
        A_B = A[:, B]
        try:
            A_B_inv = np.linalg.inv(A_B)
        except np.linalg.LinAlgError:
            return "Метод не может быть применён."
        
        c_B = c[B]
        
        # Шаг 3. Вычисляем базисный допустимый план двойственной задачи: y^T = c_B^T * A_B_inv.
        y = A_B_inv.T @ c_B  
        
        # Шаг 4. Вычисляем псевдоплан для прямой задачи.
        kappa_B = A_B_inv @ b
        kappa = np.zeros(n)
        for idx, basis_index in enumerate(B):
            kappa[basis_index] = kappa_B[idx]
        
        # Если все компоненты псевдоплана неотрицательны, мы нашли оптимальное решение.
        if np.all(kappa >= -tol):
            return kappa
        
        # Шаг 6. Находим базисный индекс, где псевдоплан отрицательный.
        for idx, basis_index in enumerate(B):
            if kappa[basis_index] < -tol:
                row_index = idx
                break
        
        delta_y = A_B_inv[row_index, :]
        
        mu = {}
        non_basis = [j for j in range(n) if j not in B]
        for j in non_basis:
            mu[j] = np.dot(delta_y, A[:, j])
        
        if all(mu[j] >= -tol for j in non_basis):
            return "Задача не совместна (не существует допустимого плана)."
        
        sigma = {}
        for j in non_basis:
            if mu[j] < -tol:
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
