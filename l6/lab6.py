import numpy as np
from typing import Tuple, List, Optional

def quadratic_programming(c: np.ndarray, D: np.ndarray, A: np.ndarray, b: np.ndarray,
                         x: np.ndarray, Jb: List[int], Jb_star: List[int], max_iter: int = 100) -> Tuple[np.ndarray, bool]:
    """
    Решает задачу квадратичного программирования методом итераций.
    
    Args:
        c: Вектор линейных коэффициентов.
        D: Симметричная положительно полуопределённая матрица.
        A: Матрица ограничений.
        b: Вектор правых частей.
        x: Начальный допустимый план.
        Jb: Опорные индексы.
        Jb_star: Расширенные опорные индексы.
        max_iter: Максимальное число итераций.
    
    Returns:
        Кортеж (оптимальный план x, флаг успеха).
    """
    n = len(c)
    iteration = 0
    
    while iteration < max_iter:
        print(f"\nИтерация {iteration + 1}")
        print(f"Текущий план: {x}")
        print(f"Jb: {Jb}, Jb*: {Jb_star}")
        
        # ШАГ 1: Вычисляем c(x), u(x), Delta(x)
        c_x = c + D @ x
        Ab = A[:, Jb]
        Ab_inv = np.linalg.inv(Ab)
        cb_x = c_x[Jb]
        u_x = -cb_x @ Ab_inv
        Delta_x = u_x @ A + c_x
        
        # print(f"c(x): {c_x}")
        # print(f"u(x): {u_x}")
        # print(f"Delta(x): {Delta_x}")
        
        # ШАГ 2: Проверяем оптимальность
        if np.all(Delta_x >= 0):
            print("Найден оптимальный план!")
            return x, True
        
        # ШАГ 3: Выбираем j0 — индекс с отрицательной компонентой Delta(x)
        j0 = np.argmin(Delta_x)
        if Delta_x[j0] >= 0:
            j0 = next(i for i, d in enumerate(Delta_x) if d < 0)
        # print(f"Выбран j0: {j0}")
        
        # ШАГ 4: Строим вектор ell
        ell = np.zeros(n)
        ell[j0] = 1  # Для неопорных компонент
        Ab_star = A[:, Jb_star]
        D_star = D[np.ix_(Jb_star, Jb_star)]
        H = np.block([[D_star, Ab_star.T], [Ab_star, np.zeros((len(Jb), len(Jb)))]])
        H_inv = np.linalg.inv(H)
        
        b_star = np.concatenate((D[Jb_star, j0], A[:, j0]))
        x_step = -H_inv @ b_star
        ell[Jb_star] = x_step[:len(Jb_star)]
        # print(f"Вектор ell: {ell}")
        
        # ШАГ 5: Вычисляем theta
        delta = ell @ D @ ell
        theta_j0 = float('inf') if delta == 0 else abs(Delta_x[j0]) / delta
        theta = {j0: theta_j0}
        
        for j in Jb:
            if ell[j] < 0:
                theta[j] = -x[j] / ell[j]
            else:
                theta[j] = float('inf')
        
        theta_0 = min(theta.values())
        j_star = min(theta, key=theta.get)
        # print(f"theta: {theta}")
        # print(f"theta_0: {theta_0}, j_star: {j_star}")
        
        if theta_0 == float('inf'):
            print("Целевая функция не ограничена снизу!")
            return x, False
        
        # ШАГ 6: Обновляем план и опоры
        x = x + theta_0 * ell
        Jb_new, Jb_star_new = Jb.copy(), Jb_star.copy()
        
        if j_star == j0:
            if j_star not in Jb_star_new:
                Jb_star_new.append(j_star)
        elif j_star in Jb_star and j_star not in Jb:
            Jb_star_new.remove(j_star)
        elif j_star in Jb:
            s = Jb.index(j_star)
            j_plus_candidates = [j for j in Jb_star if j not in Jb]
            found = False
            for j_plus in j_plus_candidates:
                if (Ab_inv @ A[:, j_plus])[s] != 0:
                    Jb_new[s] = j_plus
                    Jb_star_new.remove(j_star)
                    found = True
                    break
            if not found:
                Jb_new[s] = j0
                Jb_star_new[Jb_star_new.index(j_star)] = j0
        
        Jb, Jb_star = Jb_new, Jb_star_new
        iteration += 1
    
    print("Достигнуто максимальное число итераций.")
    return x, False


if __name__ == '__main__':
    c = np.array([-8, -6, -4, -6])  # коэффициенты линейной части из нашей целевой функции
    D = np.array([[2, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])  # матрица квадратичной части
    A = np.array([[1, 0, 2, 1], [0, 1, -1, 2]])  # матрица ограничений
    b = np.array([2, 3])  # правые части линейных уравнений
    x0 = np.array([2, 3, 0, 0]) 
    Jb0 = [0, 1]  # опора ограничений
    Jb_star0 = [0, 1]  # расширенная опора ограничений

    result, success = quadratic_programming(c, D, A, b, x0, Jb0, Jb_star0)
    print(f"\nРезультат: {result}")
    print(f"Успех: {success}")