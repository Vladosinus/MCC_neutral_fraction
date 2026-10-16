import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from initiate_constants import *
import auxillary_recombination

## Создаем сетку в сечении
def create_grid(a_beam, b_beam):
    la = np.linspace(-a_beam/2, a_beam/2, 10)
    lb = np.linspace(-b_beam/2, b_beam/2, 10)
    return la, lb



def run_MCC():
    
    ## Создаем сетку в сечении
    a_grid, b_grid = create_grid(a1, b1)
    # Создаем набор сечений
    l_grid = np.linspace(0, L, section_amount)

    ## Распределеяем частицы в пределах одного сечения, в каждйю ячейку, параллельно
    y_cell, z_cell = auxillary_recombination.distribute_particles(a_grid, b_grid, N)
    ## Назначаем частицам скорости и угол
    mean_energy, mean_angle = auxillary_recombination.section_treater()
    vx0, v_transverse = auxillary_recombination.calculate_velocity(mean_energy, mean_angle)
    # Назначаем скорости по азимутальному углу, тоже параллельно
    vy0, vz0 = auxillary_recombination.distribute_velocity_projections(a_grid, b_grid, v_transverse)
    ## Запускаем частицы
    time, particles_trajectoies, collisions_count, collision_points = auxillary_recombination.trace_particles(L, a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0)
    # Все переменные выше имеют размерность:
    # ячейки по короткой стороне, ячейки под длинной стороне, число частиц внутри ячейки. В каждой ячейке уже лежит всякое

    # Сохранение в файл
    with open('MCC_traj.txt', 'w', encoding = 'utf-8') as f:
        f.write('section 0\n')
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


if __name__ == '__main__':
    calc_type = 'MCC'
    if calc_type == 'MCC':
        run_MCC()
    elif calc_type == 'concentration':
        print(1)
