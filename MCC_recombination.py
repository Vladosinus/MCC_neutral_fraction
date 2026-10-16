import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

from initiate_constants import *
import auxillary_recombination

## Создаем сетку в сечении
def create_grid(a_beam, b_beam):
    la = np.linspace(-a_beam/2, a_beam/2, 10)
    lb = np.linspace(-b_beam/2, b_beam/2, 10)
    return la, lb



def run_MCC_atoms():
    """
    Считает движение атомов с помощью Монте-Карло, учитывает возможную рекомбинацию на стенке.
    Выводит массив всех переменных с флагом о том, что частица рекомбинировала, и координаты точки, где произошла рекомбинация
        """

    ## Создаем сетку в сечении
    a_grid, b_grid = create_grid(a1, b1)
    # Создаем набор сечений
    l_grid = np.linspace(0, L, section_amount)

    ## Назначаем частицам скорости и угол
    mean_energy, mean_angle = auxillary_recombination.section_treater()
    # Проекции скорости
    vx0, v_transverse = auxillary_recombination.calculate_velocity(mean_energy, mean_angle)
    
    for k in range(len(l_grid)):
        ## Распределеяем частицы в пределах одного сечения, в каждйю ячейку, параллельно
        y_cell, z_cell = auxillary_recombination.distribute_particles(a_grid, b_grid, N)
        # Назначаем скорости по азимутальному углу, тоже параллельно
        vy0, vz0 = auxillary_recombination.distribute_velocity_projections(a_grid, b_grid, v_transverse)
        ## Запускаем частицы, получаем результаты для одного сечения
        time, particles_trajectoies, collisions_count, collision_points, is_recombined = auxillary_recombination.trace_atoms(L, l_grid[k], a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0, recombination_probability)
        # Все переменные выше имеют размерность:
        # ячейки по короткой стороне, ячейки под длинной стороне, число частиц внутри ячейки. В каждой ячейке уже лежит всякое

        # Сохранение в файл
        # Если файл существует с прошлой итерации, то дописываем, если не существует, то создаем
        if os.path.exists('MCC_traj.txt'):
            mode = 'a'
        else:
            mode = 'w'
        with open(f'MCC_traj_{recombination_probability}.txt', mode, encoding = 'utf-8') as f:
            f.write(f'section {l_grid[k]}, м\n')
            for i in range(len(a_grid)):
                for j in range(len(b_grid)):
                    f.write(f'cell indexes {i}, {j}\n')
                    f.write(f'time\n')
                    f.write(f'{time[i, j]}\n')
                    f.write(f'trajectories\n')
                    f.write(f'{particles_trajectoies[i, j]}\n')
                    f.write(f'collisions_count\n')
                    f.write(f'{collisions_count[i, j]}\n')
                    f.write(f'collision_points\n')
                    f.write(f'{collision_points[i, j]}\n')
                    f.write(f'is_recombined\n')
                    f.write(f'{is_recombined[i, j]}\n')

# После сохранения вывода из атомов, нужно его прочитать и пееформатировать под ввод в начальные условия движения молекул, затем снова сохранить





def run_MCC_molecules():
    """Считает траектории молекул после рекомбинации,
    так как молекулы появляются в произвольных местах,
    нет разделения по сечениям, есть только массив скоростей и координат старта движения.
    """
    # Загружаем координаты молекул

    


    


if __name__ == '__main__':
    calc_type = 'MCC_atoms'
    if calc_type == 'MCC_atoms':
        run_MCC_atoms()
    elif calc_type == 'concentration':
        print(1)
