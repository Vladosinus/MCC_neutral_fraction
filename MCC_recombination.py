from operator import is_

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

    # Удаляем файл на всякий случай, если он есьт
    if os.path.exists(f'MCC_traj_{recombination_probability}.h5'):
        os.remove(f'MCC_traj_{recombination_probability}.h5')
    
    for k in range(len(l_grid)):
        ## Распределеяем частицы в пределах одного сечения, в каждйю ячейку, параллельно
        y_cell, z_cell = auxillary_recombination.distribute_particles(a_grid, b_grid, N)
        # Назначаем скорости по азимутальному углу, тоже параллельно
        vx0, vy0, vz0 = auxillary_recombination.distribute_velocity_projections(a_grid, b_grid, mean_energy)

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
    
def h5py_atoms_concentration_saver(filename, l_counter, sum_part, dt_regular, particle_count):

    with h5py.File(filename, 'w') as f:
        # Координаты сечений
        dset_l = f.create_dataset('l_counter', shape = (len(l_counter)), dtype = np.float32)
        # Количество частиц
        dset_part = f.create_dataset('sum_part', shape = (len(sum_part), len(l_counter)), dtype = np.float32)
        # Шаг по времени
        dset_timstep = f.create_dataset('timestep', shape = (1), dtype = np.float32)
        # Шаг по времени
        dset_particle_count = f.create_dataset('particle_count', shape = (1), dtype = np.float32)
        
        for i in range(len(l_counter)):
            dset_l[i] = l_counter[i]
        
        for s in range(len(sum_part)):
            for i in range(len(l_counter)):
                dset_part[s, i] = sum_part[s, i]

        dset_timstep[0] = dt_regular
        dset_particle_count[0] = particle_count
    
if __name__ == '__main__':

    # Имя файла
    filename_atoms = f'MCC_atoms_traj_{recombination_probability}.h5'
    filename_molecules = f'MCC_molecules_traj_{recombination_probability}.h5'
    # ==============================================
    # === Атомы ===
    # Запуск расчета Монте-Карло для атомов
    # run_MCC_atoms(filename_atoms)

    # Подсчет концентрации атомов с нормировкой
    # l_counter, sum_part, dt_regular, particle_count_initial = auxillary_recombination.calc_dens_atoms_in_time(filename_atoms)
    
    # Сохранение полученных распределений в файл
    # h5py_atoms_concentration_saver('atoms_conc.h5', l_counter, sum_part, dt_regular, particle_count_initial)
    
    # Чтение распределений из файла
    with h5py.File('atoms_conc.h5', 'r') as f:
        l_grid = f[f'l_counter'][:]
        part_count_in_section = f[f'sum_part'][:]
        timestep = f['timestep'][0]
        particle_count_initial = int(f['particle_count'][0])
    dl = l_grid[1] - l_grid[0]

    # Загружаем сечения процессов электронами
    nrg, velocity, sigma_electron_0, sigma_electron_1, \
                sigma_electron_2, sigma_electron_3, sigma_electron_4, \
                    sigma_electron_0_vel, sigma_electron_1_vel, sigma_electron_2_vel, \
                        sigma_electron_3_vel, sigma_electron_4_vel = auxillary_recombination.load_electron_sigmas()

    # Считаем sigma*v для электронов
    Te = 7
    sigma_e_V_0, sigma_e_V_1, sigma_e_V_2, sigma_e_V_3, sigma_e_V_4 = auxillary_recombination.calc_sigmaV(Te, velocity,
                                                                                                            sigma_electron_0_vel,
                                                                                                            sigma_electron_1_vel,
                                                                                                            sigma_electron_2_vel,
                                                                                                            sigma_electron_3_vel,
                                                                                                            sigma_electron_4_vel)
    
    # ====================================================
    # Распределение газа в покое

    # Коэффициент диффузии газа
    D_gas = 1/3*((3*kB*T/M2)**0.5)*Dh*k
    # Поток в нейтрализатор
    inflow = Q/(kB*T*a*b)

    vars = l_grid, dl, D_gas, inflow
    solution = auxillary_recombination.calculate_diffusion_flow(vars)
    n_gas = solution.y[:, -1]
    n_gas_still = n_gas

    # Ниже должен быть цикл
    iter_number = 0
    while iter_number < 5:
        # Компонентный состав пучка
        I = auxillary_recombination.beam_transform(n_gas, l_grid)
        n_beam = I/(qe*v_beam*a1*b1)

        # Генерация пучком
        beam_properties = v_beam, n_beam
        Q = auxillary_recombination.generation_by_beam(n_gas, beam_properties)
        Q_total = np.sum(Q, axis = 0)

        # Новое распределение газа с учетом потерь из-за пучка
        vars = l_grid, dl, D_gas, inflow, n_beam, v_beam, n_gas
        solution = auxillary_recombination.calculate_diffusion_flow_with_beam(vars)
        n_gas = solution.y[:, -1]

        iter_number += 1
    # На выходе из цикла имеем распределение газа, обхемную генерацию в сечениях

    # Найдем количество частиц настоящих, которые рождаютс яза timestep
    N_real = np.zeros_like(l_grid)
    for s in range(len(l_grid)):
        if s == 0 or s == len(l_grid) - 1:
            N_real[s] = (dl/2)*a1*b1*timestep*Q_total[s]
        else:
            N_real[s] = dl*a1*b1*timestep*Q_total[s]
    # В каждом сечении в единицу времени рождается тоже известное количество частиц, тогда стоимость одной частицы в сечении
    weight = N_real/particle_count_initial

    # Концентрация
    concentration = np.zeros((len(l_grid), len(l_grid)))
    for s in range(len(l_grid)): # Цикл по сечениям рождения
        for j in range(len(l_grid)): # Цикл по сечениям подсчета
            if j == 0 or j == len(l_grid) - 1:
                concentration[s, j] = part_count_in_section[s, j]*weight[s]/((dl/2)*a*b)
            else:
                concentration[s, j] = part_count_in_section[s, j]*weight[s]/(dl*a*b)

    total_atoms_concentration = np.sum(concentration, axis = 0)

    # plt.plot(l_grid, total_atoms_concentration)
    # plt.show()

    # Проверим на законе диффузии, возьмем из первого сечения генерацию, по среднему времени жизни найдем скорость дрефа
    Q_diff = Q_total[0]

    a_grid, b_grid, l_grid, time, particles_trajectoies, _, _, is_recombined = auxillary_recombination.h5py_atoms_loader('MCC_atoms_traj_0.1.h5')

    in_cell = []
    for i in range(len(a_grid)):
        for j in range(len(b_grid)):
            for n in range(len(time[0][i][j])):
                if not is_recombined[0][i][j][n]:
                    in_cell.append((time[0][i][j][n][-1]))

    v_diff = l_grid[-1]/np.mean(in_cell)       
                
    D_diff = 1/3*v_diff*Dh*k
    l_grid1 = np.linspace(0, L, 50)
    dl1 = l_grid1[1] - l_grid1[0]
    vars = l_grid1, dl1, D_diff, Q_diff

    # Функция для расчета концентрации
    def calculate_diffusion_flow(vars):
        l, dl, D, Q = vars

        # Просто натекание из левой границы, поток
        t_pass = L/((3*kB*T/M2)**0.5)
        t_max = 50*t_pass

        def diffusion(t, n):
            dndt = np.zeros_like(n)
            for i in range(1, len(dndt) - 1):
                d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
                dndt[i] = D*d2n_dx2

            # dndt[0] = 2*D*(n[1] - n[0])/dl**2 + Q
            dndt[0] = D*(n[0] - 2*n[0] + n[1])/dl**2 + Q
            # dndt[0] = D*(n[0] - 2*n[1] + n[2])/dl**2 + Q
            # dndt[0] = D*(2*n[0] - 5*n[1] + 4*n[2] - n[3])/dl**2 + Q
            dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2

            return dndt
        
        n_IC = np.zeros(len(l))
        solution = solve_ivp(fun=diffusion,
                            t_span=(0, t_max),
                            y0=n_IC,
                            method='RK45')
            
        return solution
    solution = calculate_diffusion_flow(vars)
    plt.plot(l_grid1, solution.y[:, -1]*(1 - 0.92))
    plt.plot(l_grid, concentration[0])
    plt.show()






























    exit()
# Запуск решателя Монте-Карло для вторичных молекул
    # run_MCC_molecules(filename_molecules, filename_atoms)

    # Подсчет концентрации молекул
    # l_grid, molecule_count_per_section = auxillary_recombination.calc_au_molecules_concentration(filename_molecules)


    






    



    





    # Подсчет концентрации атомов без нормировки
    # l_grid, atom_count_per_section = auxillary_recombination.calc_au_atoms_concentration(filename_atoms)




















    
    