import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import h5py
import math
from scipy.interpolate import interp1d

from initiate_constants import *
import auxillary_recombination
import trace_atoms_parallel
import trace_molecules_parallel

# Монте-Карло расчет движения вторичных атомов
def run_MCC_atoms(filename):
    """
    Считает движение атомов с помощью Монте-Карло, учитывает возможную рекомбинацию на стенке.
    Выводит массив всех переменных с флагом о том, что частица рекомбинировала, и координаты точки, где произошла рекомбинация
        """

    # Формат данных внутри файла, для хранения массивов разной длины
    dt = h5py.vlen_dtype(np.dtype('float64'))

    ## Создаем сетку в сечении
    a_grid, b_grid, l_grid = auxillary_recombination.create_grid(a1, b1)

    ## Назначаем частицам скорости и угол
    mean_energy, mean_angle = auxillary_recombination.section_treater()
    # Проекции скорости
    vx0, v_transverse = auxillary_recombination.calculate_velocity(mean_energy, mean_angle)

    # Удаляем файл на всякий случай, если он есьт
    if os.path.exists(f'MCC_traj_{recombination_probability}.h5'):
        os.remove(f'MCC_traj_{recombination_probability}.h5')
    
    for k in range(len(l_grid)):
        ## Распределеяем частицы в пределах одного сечения, в каждйю ячейку, параллельно
        y_cell, z_cell = auxillary_recombination.distribute_particles(a_grid, b_grid, N)
        # Назначаем скорости по азимутальному углу, тоже параллельно
        vy0, vz0 = auxillary_recombination.distribute_velocity_projections(a_grid, b_grid, v_transverse)
        ## Запускаем частицы, получаем результаты для одного сечения
        time, particles_trajectoies, collisions_count, collision_points, is_recombined = trace_atoms_parallel.trace_atoms(L, l_grid[k], a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0, recombination_probability)
        # Все переменные выше имеют размерность:
        # ячейки по короткой стороне, ячейки под длинной стороне, число частиц внутри ячейки. В каждой ячейке уже лежит всякое
        
        # Вызываем функция для сохранения данных в файл
        # filename = f'MCC_traj_{recombination_probability}.h5'
        auxillary_recombination.h5py_atoms_saver(l_grid, k, filename, a_grid, b_grid, dt, time, particles_trajectoies, collisions_count, collision_points, is_recombined)

# Монте-Карло расчет движения молекул после рекомбинации на стенке
def run_MCC_molecules(filename_molecules, filename_atoms):
    """Считает траектории молекул после рекомбинации,
    так как молекулы появляются в произвольных местах,
    нет разделения по сечениям, есть только массив скоростей и координат старта движения.
    """

    # Формат данных внутри файла, для хранения массивов разной длины
    dt = h5py.vlen_dtype(np.dtype('float64'))

    # Загружаем координаты молекул
    a_grid, b_grid, l_grid_file, time, particles_trajectoies, collisions_count, collision_points, is_recombined = auxillary_recombination.h5py_atoms_loader(filename_atoms)
    # Число сечений
    section_count = len(time)
    # Переменные под координаты старта
    x_mol = []
    vx_mol = []
    y_mol = []
    vy_mol = []
    z_mol = []
    vz_mol = []

    # Получаем начальные условия для старта молекул
    # Идем по сечениям

    for s in range(section_count):
        # В каждом сечении проходим по всем ячейкам
        for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                # Проходим по всем частицам в сечении
                for n in range(len(time[s][i, j])):
                    if is_recombined[s][i, j, n]:
                        x_mol.append(particles_trajectoies[s][i, j, n][-1, 0])
                        vx_mol.append(particles_trajectoies[s][i, j, n][-1, 1])
                        y_mol.append(particles_trajectoies[s][i, j, n][-1, 2])
                        vy_mol.append(particles_trajectoies[s][i, j, n][-1, 3])
                        z_mol.append(particles_trajectoies[s][i, j, n][-1, 4])
                        vz_mol.append(particles_trajectoies[s][i, j, n][-1, 5])
    # Конвертируем в массивы из списков
    x_mol = np.array(x_mol)
    vx_mol = np.array(vx_mol)
    y_mol = np.array(y_mol)
    vy_mol = np.array(vy_mol)
    z_mol = np.array(z_mol)
    vz_mol = np.array(vz_mol)
    # Так как частиц мало для плавной кривой, увеличим искусственно их количество
    # Желаемое число частиц
    N_wanted = 10e3
    test = False
    if not test:
        if len(x_mol) < N_wanted:
            # Сколько раз надо увеличить
            r = math.ceil(N_wanted/len(x_mol))
            # Само увеличение
            x_mol = np.tile(x_mol, r)
            vx_mol = np.tile(vx_mol, r)
            y_mol = np.tile(y_mol, r)
            vy_mol = np.tile(vy_mol, r)
            z_mol = np.tile(z_mol, r)
            vz_mol = np.tile(vz_mol, r)

    # Вызываем организатор для параллельного расчета
    time, particles_trajectoies, collisions_count = trace_molecules_parallel.trace_molecules(L, N, x_mol, vx_mol, y_mol, vy_mol, z_mol, vz_mol)
    # Сохранение в файл
    auxillary_recombination.h5py_molecules_saver(filename_molecules, time, particles_trajectoies, collisions_count, dt)
    
    

    
if __name__ == '__main__':

    # Имя файла
    filename_atoms = f'MCC_atoms_traj_{recombination_probability}.h5'
    filename_molecules = f'MCC_molecules_traj_{recombination_probability}.h5'

    # Запуск расчета Монте-Карло для атомов
    # run_MCC_atoms(filename_atoms)

    # Подсчет концентрации атомов без нормировки
    # l_grid, atom_count_per_section = auxillary_recombination.calc_au_atoms_concentration(filename_atoms)

    # Подсчет концентрации атомов с нормировкой
    auxillary_recombination.calc_dens_atoms_in_time(filename_atoms)

    # exit()
    
    # Запуск решателя Монте-Карло для вторичных молекул
    # run_MCC_molecules(filename_molecules, filename_atoms)

    # Подсчет концентрации молекул
    # l_grid, molecule_count_per_section = auxillary_recombination.calc_au_molecules_concentration(filename_molecules)

    
    