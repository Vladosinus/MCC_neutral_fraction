import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def generate_pairs(a1, a2, b1, b2, N):
    """N пар (x, y): x ∈ [a1, a2], y ∈ [b1, b2]."""
    rng = np.random.default_rng()
    x = rng.uniform(a1, a2, N)
    y = rng.uniform(b1, b2, N)
    return np.column_stack((x, y))


if __name__ == '__main__':
    M = 1000

    # Параметры для каждой из M задач
    a1_list = np.full(M, 0.0)
    a2_list = np.full(M, 10.0)
    b1_list = np.full(M, -5.0)
    b2_list = np.full(M, 5.0)
    N_list  = repeat(M)

    workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            generate_pairs,
            a1_list, a2_list, b1_list, b2_list, N_list,
            chunksize=max(1, M // (workers * 4))
        ))

    print(f"Задач: {len(results)}")
    print(f"Форма первой: {results[0].shape}")
    total_numbers = sum(r.size for r in results)
    print(f"Всего чисел: {total_numbers}")