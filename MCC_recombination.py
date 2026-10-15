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



def run():
    
    ## Создаем сетку в сечении
    a_grid, b_grid = create_grid(a1, b1)

    ## Распределеяем частицы в пределах одного сечения, в каждйю ячейку, параллельно
    y_cell, z_cell = auxillary_recombination.distribute_particles(a_grid, b_grid, N)
    ## Назначаем частицам скорости и угол
    mean_energy, mean_angle = auxillary_recombination.section_treater()
    vx0, v_transverse = auxillary_recombination.calculate_velocity(mean_energy, mean_angle)
    # Назначаем скорости по азимутальному углу, тоже параллельно
    vy0, vz0 = auxillary_recombination.distribute_velocity_projections(a_grid, b_grid, v_transverse)
    ## Запускаем частицы
    auxillary_recombination.trace_particles(L, a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0)



if __name__ == '__main__':
    run()