import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import h5py

from initiate_constants import *

# Функция для создания сетки в сечениях
# ==========================================================================
def create_grid(a_beam, b_beam):
    la = np.linspace(-a_beam/2, a_beam/2, 5)
    lb = np.linspace(-b_beam/2, b_beam/2, 5)
    l_x = np.linspace(0, L, section_amount)
    return la, lb, l_x

# Функция для распределения частиц по ячейкам в заданном количестве
# ==========================================================================
# Сам генератор
def _generate_cell(args):
    # Распаковываем аругменты
    i, j, la_left, la_right, lb_bottom, lb_top, N = args
    rng = np.random.default_rng()  # or pass a seed for reproducibility

    # генерируем случайный координаты внутри ячеек
    a_cell = rng.uniform(la_left, la_right, N)
    b_cell = rng.uniform(lb_bottom, lb_top, N)

    return i, j, a_cell, b_cell
# Организатор
def distribute_particles(la, lb, N):

    print('Генерация стартовых координат частиц...')

    # Чисол процессов
    n_workers = os.cpu_count()
    tasks = []
    # Создаем пул задач
    for i in range(len(la) - 1):
        for j in range(len(lb) - 1):
            tasks.append((i, j, la[i], la[i+1], lb[j], lb[j+1], N))
    # Вызов параллельного решателя
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_generate_cell, tasks))
    # Заготовки под массивы координат
    a_cell = np.zeros((len(la), len(lb), N))
    b_cell = np.zeros((len(la), len(lb), N))
    # Распаковываем результат
    for i, j, a_slice, b_slice in results:
        a_cell[i, j, :] = a_slice
        b_cell[i, j, :] = b_slice

    return a_cell, b_cell
# ==========================================================================

## Определение средней энергии и среднего угла вторичных частиц
def section_treater():
    # Здесь будет найдена средняя энергия частиц и средний угол вылета, чтобы потом их отправить в calculate_velocity
    
    secondary_energy = np.linspace(0, 20, 100)
    ## 10 градусов
    theta10 = np.loadtxt(testprofile1)
    nrg1 = theta10[:, 0]
    sig = theta10[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=15, polyorder=3)
    # Интерполяция на общую сетку
    theta10_interpolator = interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma10 = theta10_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma10[sigma10 < 0] = 0

    ## 50 градусов
    theta50 = np.loadtxt(testprofile2)
    nrg1 = theta50[:, 0]
    sig = theta50[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=11, polyorder=3)
    # Интерполяция на общую сетку
    theta50_interpolator = interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma50 = theta50_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma50[sigma50 < 0] = 0

    ## 90 градусов
    theta90 = np.loadtxt(testprofile3)
    nrg1 = theta90[:, 0]
    sig = theta90[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=11, polyorder=3)
    # Интерполяция на общую сетку
    theta90_interpolator = interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma90 = theta90_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma90[sigma90 < 0] = 0,

    ## 0 градусов
    # Искусственно создадим такое распрееление, так как измнрить его невозможно
    sigma0 = sigma10*1.05

    thetas = [0, 10, 50, 90]

    sigmas = np.column_stack((sigma0, sigma10, sigma50, sigma90))
    
    ## Средняя энергия
    only_energy_distibution = np.zeros(len(sigmas[:, 0]))
    for i in range(len(sigmas[:, 0])):
        only_energy_distibution[i] = np.trapezoid(sigmas[i, :], thetas)

    ## Средний угол
    only_angle_distibution = np.zeros(len(sigmas[0, :]))
    for i in range(len(sigmas[0, :])):
        only_angle_distibution[i] = np.trapezoid(sigmas[:, i], secondary_energy)

    #  Находим средние значения энергии и угла
    mean_energy = np.trapezoid(only_energy_distibution*secondary_energy, secondary_energy)/np.trapezoid(only_energy_distibution, secondary_energy)
    mean_angle = np.trapezoid(only_angle_distibution*thetas, thetas)/np.trapezoid(only_angle_distibution, thetas)

    return mean_energy, mean_angle

## Распределяем скорости по азимутальному углу
# ==========================================================================
# Генератор заданий
def _velocity_projections(args):
    i, j, N, mean_energy = args
    vx0  = np.zeros(N)
    vy0 = np.zeros(N)
    vz0 = np.zeros(N)

    # Модуль скорости
    speed = (2*mean_energy*qe/M)**0.5
    
    for n in range(N):
        phi = np.random.rand()*90
        vx0[n] = speed*np.abs(np.cos(np.deg2rad(phi)))
        v_transverse = speed*np.abs(np.sin(np.deg2rad(phi)))

        gamma = np.random.rand()*360
        if gamma >= 0 and gamma < 90:
            vy0[n] = v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = -v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 90 and gamma < 180:
            vy0[n] = -v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = -v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 180 and gamma < 270:
            vy0[n] = -v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 270 and gamma < 360:
            vy0[n] = v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
    
    return i, j, vx0, vy0, vz0

# Организатор
def distribute_velocity_projections(la, lb, mean_energy):
    print('Назначение исходных скоростей...')
    # Число потоков
    n_workers = os.cpu_count()
    # Пул задач
    tasks = []
    for i in range(len(la)):
        for j in range(len(lb)):
            tasks.append((i, j, N, mean_energy))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_velocity_projections, tasks))

    vel_x = np.zeros((len(la), len(lb), N))
    vel_y = np.zeros((len(la), len(lb), N))
    vel_z = np.zeros((len(la), len(lb), N))
    # распаковываем результат
    for i, j, vx_slice, vy_slice, vz_slice in results:
        vel_x[i, j, :] = vx_slice
        vel_y[i, j, :] = vy_slice
        vel_z[i, j, :] = vz_slice

    return vel_x, vel_y, vel_z

# ==========================================================================
# Скорости при диффузном отражении
def diffuse_reflection(vars, wall_idx, particle_type):
    """Диффузное отражение частицы от стенки/входного сечения.

    Модуль скорости сохраняется, направление разыгрывается случайно в
    полусфере внутренней нормали стенки.

    wall_idx:
        1 - левая стенка  (y = -a/2), внутренняя нормаль +y
        2 - правая стенка (y = +a/2), внутренняя нормаль -y
        3 - нижняя стенка (z = -b/2), внутренняя нормаль +z
        4 - верхняя стенка(z = +b/2), внутренняя нормаль -z
        5 - входное сечение (x = 0), внутренняя нормаль +x

    Возвращает новый список скоростей [vx, vy, vz].
    """
    # Потери при соударении со стенкой 10,2 % от разницы энергии стенки и частицы
    if particle_type == 'atoms':
        # Если атом, обмен энергией со стенкой
        speed_old = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
        nrg = (M*speed_old**2)/2
        deltaE = nrg/qe - 300/11604
        loss = 0.102*deltaE
        nrg_new = nrg/qe - loss
        speed = (2*nrg_new*qe/M)**0.5

    elif particle_type == 'molecules':
        # Если молекула, обмен энергией со стенкой
        speed_old = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
        nrg = (M*speed_old**2)/2
        deltaE = nrg/qe - 300/11604
        loss_collision = 0.102*deltaE
        add_by_energy = 0.5*deltaE
        nrg_new = mean_angle_molecule_nrg - loss_collision + add_by_energy
        speed = (2*nrg_new*qe/2/M)**0.5


    # Полярный угол от внутренней нормали и азимутальный угол
    theta = np.deg2rad(np.random.rand()*180)
    v_normal = speed*np.sin(theta)      # компонента вдоль внутренней нормали (>= 0)
    v_tang = speed*np.cos(theta)
    phi = np.deg2rad(np.random.rand()*360)
    t1 = v_tang*np.cos(phi)
    t2 = v_tang*np.sin(phi)

    if wall_idx == 1:      # левая, нормаль +y
        return [t1, v_normal, t2]
    elif wall_idx == 2:    # правая, нормаль -y
        return [t1, -v_normal, t2]
    elif wall_idx == 3:    # нижняя, нормаль +z
        return [t1, t2, v_normal]
    elif wall_idx == 4:    # верхняя, нормаль -z
        return [t1, t2, -v_normal]
    elif wall_idx == 5:    # вход, нормаль +x
        return [v_normal, t1, t2]
    else:
        return [vars[1], vars[3], vars[5]]

# ==========================================================================
# Функция для сохранения MCC атомов в файл
def h5py_atoms_saver(l_grid, k, filename, a_grid, b_grid, dt, time, particles_trajectoies, collisions_count, collision_points, is_recombined):

    # Если в пределах запуска файл существует, до в него дописываем,
    # если нет, то создаем и пишем для следующих сечений в него
    if os.path.exists(filename):
        mode = 'a'
    else:
        mode = 'w'

    # Сама запись в файл
    with h5py.File(filename, mode) as f:

        # Создание группы - сечение
        grp = f.create_group(f'section_{k}')

        # Заготовка датасетов под переменные
        # Время
        dset_time = grp.create_dataset('time', shape = (len(a_grid), len(b_grid), N), dtype=dt)
        # Траектории
        dset_particles_trajectoies_data = grp.create_dataset('particles_trajectoies_data', shape = (len(a_grid), len(b_grid), N), dtype=dt)
        dset_particles_trajectoies_shape = grp.create_dataset('particles_trajectoies_shape', shape = (len(a_grid), len(b_grid), N, 2), dtype=np.int32)
        # Количество соударений
        dset_collisions_count = grp.create_dataset('collisions_count', shape = (len(a_grid), len(b_grid), N), dtype=np.int32)
        # # Точки соударений со стенкой
        dset_collision_points_data = grp.create_dataset('collision_points_data', shape = (len(a_grid), len(b_grid), N), dtype=dt)
        dset_collision_points_shape = grp.create_dataset('collision_points_shape', shape = (len(a_grid), len(b_grid), N, 2), dtype=np.int32)
        # Флаг, была ли рекомбинация
        dset_is_recombined = grp.create_dataset('is_recombined', shape = (len(a_grid), len(b_grid), N), dtype=np.int32)

        # Заполнение датасетов
        for i_h5 in range(len(a_grid)):
            for j_h5 in range(len(b_grid)):
                for k_h5 in range(N):
                    # Время жизни
                    dset_time[i_h5, j_h5, k_h5] = time[i_h5][j_h5][k_h5]
                    # Количество соударений
                    dset_collisions_count[i_h5, j_h5, k_h5] = collisions_count[i_h5][j_h5][k_h5]
                    # Флаг рекомбинации
                    dset_is_recombined[i_h5, j_h5, k_h5] = is_recombined[i_h5, j_h5, k_h5]
    
        for i_h5 in range(len(a_grid)):
            for j_h5 in range(len(b_grid)):
                for k_h5 in range(N):
                    # Траектории
                    arr = particles_trajectoies[i_h5, j_h5, k_h5]
                    dset_particles_trajectoies_data[i_h5, j_h5, k_h5] = arr.ravel()
                    dset_particles_trajectoies_shape[i_h5, j_h5, k_h5] = arr.shape
                    # Точки столкновений
                    arr = collision_points[i_h5, j_h5, k_h5]
                    dset_collision_points_data[i_h5, j_h5, k_h5] = arr.ravel()
                    dset_collision_points_shape[i_h5, j_h5, k_h5] = arr.shape

        if k == len(l_grid) - 1:
            grp = f.create_group(f'geometry')
            dset_geometry = grp.create_dataset('geometry', shape = (3, 1), dtype = dt)
            dset_geometry[0] = a_grid
            dset_geometry[1] = b_grid
            dset_geometry[2] = l_grid

# ==========================================================================
# Функция для чтения MCC атомов из файла
def h5py_atoms_loader(filename):
    # Открываем файл
    with h5py.File(filename, 'r') as f:

        # Чтение геометрии, тодельно, чтобы не в цикле
        grp_geo = f['geometry']
        geom = grp_geo['geometry'][:]
        # Ниже второй [0] нужен, так как это список массивов с одним массивм
        a_grid = geom[0][0]
        b_grid = geom[1][0]
        l_grid = geom[2][0]

        # Заготовки под траектории и положения столкновений

        time = np.empty(( len(f.keys()) - 1), dtype=object)
        is_recombined = np.empty(( len(f.keys()) - 1), dtype=object)
        # Так как они многомерные, и для них использовалась форма
        particles_trajectoies = np.empty(( (len(f.keys()) - 1), len(a_grid), len(b_grid), N), dtype = object)
        collision_points = np.empty(( (len(f.keys()) - 1), len(a_grid), len(b_grid), N), dtype = object)
        
        # Так как сечений много, будет цикл по сечениям
        for k in range(len(f.keys()) - 1):
            
            # Чтение времени, там одномерные массивы
            grp = f[f'section_{k}']
            # Время
            time[k] = grp['time'][:]
            # Траектории
            particles_trajectoies_data = grp['particles_trajectoies_data'][:]
            particles_trajectoies_shape = grp['particles_trajectoies_shape'][:]
            # Количество соударений
            collisions_count = grp['collisions_count'][:]
            # Точки соударений        
            collision_points_data = grp['particles_trajectoies_data'][:]
            collision_points_shape = grp['particles_trajectoies_shape'][:]
            # Флаг рекомбинации
            is_recombined[k] = grp['is_recombined'][:]
            
            for i in range(len(a_grid)):
                for j in range(len(b_grid)):
                    for n in range(N):
                        # Траектории
                        M, K = particles_trajectoies_shape[i, j, n]                            
                        particles_trajectoies[k, i, j, n] = particles_trajectoies_data[i, j, n].reshape(M, K)
                        # Точки столкновений
                        M, K = collision_points_shape[i, j, n]
                        collision_points[k, i, j, n] = collision_points_data[i, j, n].reshape(M, K)

    return a_grid, b_grid, l_grid, time, particles_trajectoies, collisions_count, collision_points, is_recombined

# ==========================================================================
# Функция для сохранения MCC молекул в файл
def h5py_molecules_saver(filename, time, particles_trajectoies, collisions_count, dt):

    with h5py.File(filename, 'w') as f:
        # Создание датасетов
        # Время
        dset_time = f.create_dataset('time', shape = (len(time)), dtype = dt)
        # Траектории
        dset_particles_trajectoies_data = f.create_dataset('particles_trajectoies_data', shape = (len(time)), dtype = dt)
        dset_particles_trajectoies_shape = f.create_dataset('particles_trajectoies_shape', shape = (len(time), 2), dtype = np.int32)
        # Количество соударений
        dset_collisions_count = f.create_dataset('collisions_count', shape = (len(time)), dtype = np.int32)
        # Сохранение
        for i in range(len(time)):
            # Время
            dset_time[i] = time[i]
            # Траектории
            arr = particles_trajectoies[i]
            dset_particles_trajectoies_data[i] = arr.ravel()
            dset_particles_trajectoies_shape[i] = arr.shape
            # Количество соударений
            dset_collisions_count[i] = collisions_count[i]
            
# ==========================================================================
# Функция для чтения MCC молекул из файла
def h5py_molecules_loader(filename):
    with h5py.File(filename, 'r') as f:

        # Время
        time = f[f'time'][:]
        # Число столкновений
        collisions_count = f[f'collisions_count'][:]

        # Траектории
        particles_trajectoies_data = f[f'particles_trajectoies_data'][:]
        particles_trajectoies_shape = f[f'particles_trajectoies_shape'][:]

        particles_trajectoies = np.empty(len(time), dtype = object)
        
        for i in range(len(time)):
            M, K = particles_trajectoies_shape[i]
            particles_trajectoies[i] = particles_trajectoies_data[i].reshape(M, K)

    return time, particles_trajectoies, collisions_count
        
# ==========================================================================
# Функция для подсчета концентрации атомов
def calc_au_atoms_concentration(filename):
    """
    Пересчитывает распределение в штуках частиц из координат на заданной сетке вдоль нейтрализатора
    """
    # Загружаем файл с концентрациями
    a_grid, b_grid, l_grid_file, time, particles_trajectoies, collisions_count, collision_points, is_recombined = h5py_atoms_loader(filename)
        
    section_count = len(time)
    l_grid = np.linspace(0, l_grid_file[-1], 50)
    hl = l_grid[1] - l_grid[0]
    
    # Пустая заготовка под накопление частиц во всех сечениях
    particles_total = np.empty((section_count, len(a_grid), len(b_grid), N), dtype = object)

    # Идем по сечениям
    for s in range(section_count):
        # В каждом сечении проходим по всем ячейкам
        for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                # Проходим по всем частицам в сечении
                for n in range(len(time[s][i, j])):
                    
                    # Пустая переменная под накопление частиц
                    part_count = np.zeros((len(l_grid), len(time[s][i, j, n])))
                    # Проходим во массиву времени
                    for t in range(len(time[s][i, j, n])):
                        # Положение рассматриваемой частицы в рассматриваемый момнет времени
                        x_part = particles_trajectoies[s][i, j, n][t, 0]
                        # Цикл по сетке по длине нейтрализатора, чтобы искаьт
                        for l in range(len(l_grid)):
                            if l == 0:    
                                if x_part < l_grid[l] + hl/2:
                                    part_count[l, t] += 1
                                    

                            elif l == len(l_grid) - 1:
                                if x_part >= l_grid[l] - hl/2 and x_part < l_grid[l]:
                                    part_count[l, t] += 1
                                    
                            else:
                                if x_part >= l_grid[l] - hl/2 and x_part < l_grid[l] + hl/2:
                                    part_count[l, t] += 1 
                    # Собираем в распределение по длине           
                    particles_total[s, i, j, n] = np.sum(part_count, axis = 1)

    # Теперь надо собрать просто в один массив, все везде просуммировать
    # Накопление в пределах одного сечения
    count_per_cell = np.zeros((len(a_grid), len(b_grid), len(l_grid)))
    # Накопление по разным сечениям
    count_per_section = np.empty((section_count, len(l_grid)))
    for s in range(section_count):
    # В каждом сечении проходим по всем ячейкам
        for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                count_per_cell[i, j] = np.sum(particles_total[s, i, j, :], axis = 0)
        count_per_section[s] = np.sum(count_per_cell, axis = (0, 1))

    return l_grid, count_per_section

# ==========================================================================
# Функция для подсчета концентрации молекул
def calc_au_molecules_concentration(filename):
    """
    Пересчитывает распределение в штуках частиц из координат на заданной сетке вдоль нейтрализатора
    """
    # Загружаем файл с концентрациями
    time, particles_trajectoies, _ = h5py_molecules_loader(filename)
    l_grid = np.linspace(0, L, 50)
    hl = l_grid[1] - l_grid[0]
    
    # Пустая заготовка под накопление частиц во всех сечениях
    particles_total = np.zeros(len(l_grid), dtype = object)
    count_per_part = np.zeros(len(time), dtype = object)

    # Проходим по всем частицам
    for n in range(len(time)):
        # Пустая переменная под накопление частиц
        part_count = np.zeros((len(l_grid), len(time[n])))
        # Проходим во массиву времени
        for t in range(len(time[n])):
            # Положение рассматриваемой частицы в рассматриваемый момнет времени
            x_part = particles_trajectoies[n][t, 0]
            # Цикл по сетке по длине нейтрализатора, чтобы искаьт
            for l in range(len(l_grid)):
                if l == 0:    
                    if x_part < l_grid[l] + hl/2:
                        part_count[l, t] += 1
                        

                elif l == len(l_grid) - 1:
                    if x_part >= l_grid[l] - hl/2 and x_part < l_grid[l]:
                        part_count[l, t] += 1
                        
                else:
                    if x_part >= l_grid[l] - hl/2 and x_part < l_grid[l] + hl/2:
                        part_count[l, t] += 1 
        # Вклад от одной частицы во все времена по сечениям
        count_per_part[n] = np.sum(part_count, axis = 1)
    # Распределение в сечениях от всех частиц
    particles_total = np.sum(count_per_part, axis = 0)

    return l_grid, particles_total

# Функция для интерполяции на регулярной временной сетке координаты x и подсчета концентрации атома    
def calc_dens_atoms_in_time(filename):

    a_grid, b_grid, l_grid, time, particles_trajectoies, _, _, _ = h5py_atoms_loader(filename)

    section_count = len(particles_trajectoies)
    particle_count_in_section = len(particles_trajectoies[0, 0, 0])
    particle_count = len(a_grid)*len(b_grid)*N
    particles = np.empty((len(particles_trajectoies), len(a_grid)*len(b_grid)*N), dtype = object)
    times = np.empty((len(particles_trajectoies), len(a_grid)*len(b_grid)*N), dtype = object)

    # Уберем распределения ячейкам, оставим просто в сечениях
    for s in range(section_count):
        # Счетчик индексов для создаваемого массива
        k = 0
        for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                for n in range(particle_count_in_section):
                    times[s, k] = time[s][i, j, n]
                    particles[s, k] = particles_trajectoies[s, i, j, n][:, 0]
                    k += 1
    # Теперь есть просто массивы x координаты для каждого сечения для всех частиц и то же самое для времени
            
    max_in_section = np.zeros(section_count)
    mean_in_section = np.zeros(section_count)
    min_in_section = np.zeros(section_count)

    # Найдем максимальное время жизни частиц из каждого сечения
    for s in range(section_count):
        max_per_particle = []
        
        for n in range(particle_count):
            max_per_particle.append(times[s, n][-1])
                            
        mean_in_section[s] = np.mean(max_per_particle) 
        max_in_section[s] = np.max(max_per_particle)
        min_in_section[s] = np.min(max_per_particle)

    # Самое большое время существования атома во всех сечениях
    time_absolute_max = np.max(max_in_section)
    time_absolute_min = np.min(max_in_section[:-1])
            
    # Регулярная сетка по времени для интерполяции
    time_regular_count = int(1e4)
    time_grid_regular = np.linspace(0, time_absolute_max, time_regular_count)
    dt_regular = time_grid_regular[1] - time_grid_regular[0]
    x_particle_regular = np.empty((section_count, particle_count), dtype = object)

    # Интерполируем координаты по сечениям
    for s in range(section_count):
        for n in range(particle_count):
            x_interpolator = interp1d(times[s, n], particles[s, n], kind = 'linear', bounds_error = False, fill_value = None)
            x_particle_regular[s, n] = x_interpolator(time_grid_regular)

    # Имеем наборы координаты на регулярной сетке с известным шагом, чтобы можно было использовать номировку
    # Подсчет частиц

    # Что ниже пока оставить, мысль о том, чтобы использовать ту же пространственную сетку, что для генерации
    # l_counter_len = 50
    # l_counter = np.linspace(0, L, l_counter_len)
    l_counter = l_grid
    l_counter_len = len(l_counter)
    hl = l_counter[1] - l_counter[0]

    part_count = np.zeros((section_count, l_counter_len, time_regular_count), dtype = object)
    # По сечениям
    for s in range(section_count):
        # По частицам
        for n in range(particle_count):
            # По времени
            for t in range(time_regular_count):
                if x_particle_regular[s, n][t] != None:
                # По сечениям нейтрализатора
                    for l in range(l_counter_len):
                        if l == 0:    
                            if x_particle_regular[s, n][t] < l_counter[l] + hl/2:
                                part_count[s, l, t] += 1
                            
                        elif l == len(l_grid) - 1:
                            if x_particle_regular[s, n][t] >= l_counter[l] - hl/2 and x_particle_regular[s, n][t] < l_counter[l]:
                                part_count[s, l, t] += 1
                                
                        else:
                            if x_particle_regular[s, n][t] >= l_counter[l] - hl/2 and x_particle_regular[s, n][t] < l_counter[l] + hl/2:
                                part_count[s, l, t] += 1 
    # Заготовка под распределения из каждого сечения 
    sum_part = np.zeros((section_count, l_counter_len))
    
    # Собираем из разных времен в массив
    for s in range(section_count - 1):
        sum_part[s] = np.sum(part_count[s, :, :], axis = 1)

    return l_counter, sum_part, dt_regular, particle_count

# Функция для интерполяции на регулярной временной сетке координаты x и подсчета концентрации атома

            
# ==========================================================================
# Все про пучок
# Загрузка сечений для взаимодействия пучка
def load_sigmas():
    numbers = []
    started = False # Маркер начала данных
    ## Загружаются сечения дли ионов
    with open('proton_sigmas.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not started:
                if line == '':
                    started = True
                continue
            if started:
                try:
                    row = [float(x) for x in line.split()]
                    numbers.append(row)
                except ValueError:
                    continue
    proton_sigmas = np.array(numbers).ravel()

    numbers = []
    started = False
    ## Загружаются сечения дли нейтралов
    with open('neutral_sigmas.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not started:
                    if line == '':
                        started = True
                    continue
                if started:
                    try:
                        row = [float(x) for x in line.split()]
                        numbers.append(row)
                    except ValueError:
                        continue
    neutral_sigmas = np.array(numbers).ravel()
    generation_sigmas = np.array([neutral_sigmas, proton_sigmas], dtype = object)
    return generation_sigmas     

# Функция для расчета концентрации
def calculate_diffusion_flow(vars):
    l, dl, D, inflow = vars

    # Просто натекание из левой границы, поток
    t_pass = L/((3*kB*T/M2)**0.5)
    t_max = 50*t_pass

    def diffusion(t, n):
        dndt = np.zeros_like(n)
        for i in range(1, len(dndt) - 1):
            d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
            dndt[i] = D*d2n_dx2

        dndt[0] = 2*D*(n[1] - n[0])/dl**2 + 2*inflow/dl
        dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2

        return dndt
    
    n_IC = np.zeros(len(l))
    solution = solve_ivp(fun=diffusion,
                        t_span=(0, t_max),
                        y0=n_IC,
                        method='RK45')
        
    return solution

# Функция для расчет ионов и нейтралов
def beam_transform(n_gas, l_grid):
    # загрузка сечений перезарядки и обдирки
    generation_sigmas = load_sigmas()
    sigmas = [generation_sigmas[0][1], generation_sigmas[1][1]]
    ## Тут будет логика расчета, и вызов решателя
    # Правая часть будет просто объявлена, чтобы не делать функцию внутри функции
    
    t_span = (0, l_grid[-1])
    # Нормировано на 1, лучше так делать, а потом уже умножать на всякое, если надо
    I0 = [0, 1]
    args = n_gas, sigmas, l_grid

    solution = solve_ivp(rbt,
                         t_span = t_span,
                         y0 = I0,
                         method = 'RK45',
                         dense_output = True,
                         args = args,
                         max_step = L/20
                         )

    I_interpolated = solution.sol(l_grid)
    
    return I_interpolated

# Правая часть для beam_transform
def rbt(t, I, n_gas_origin, sigmas, l):

    gas_interpolator = interp1d(l, n_gas_origin, fill_value=0, bounds_error=False, kind = 'linear')
    n_gas = gas_interpolator(t)

    sigma01 = sigmas[0]
    sigma10 = sigmas[1]    

    dI0 = n_gas*I[1]*sigma10 - n_gas*I[0]*sigma01
    dI1 = n_gas*I[0]*sigma01 - n_gas*I[1]*sigma10

    return [dI0, dI1]

# Функция для расчета генерации АТОМОВ пучком и электронами       
def generation_by_beam(n_gas, beam_properties):
    # Загрузка сечений дли взаимодействия пучка
    generation_sigmas = load_sigmas()
    v, n_beam = beam_properties

    ## Генерация атомами
    neutral_total_sigma = generation_sigmas[0][2] + 2*generation_sigmas[0][3] + generation_sigmas[0][6] + generation_sigmas[0][8]
    Q0 = n_gas*n_beam[0]*neutral_total_sigma*v
    
    ## Генерация ионами
    proton_total_sigma = 2*generation_sigmas[1][3] + generation_sigmas[1][5]
    Q1 = n_gas*n_beam[1]*proton_total_sigma*v

    ## генерация вторичной плазмой
    Qe = np.zeros(len(n_gas))
    
    return [Q0, Q1, Qe]

# Функция для расчета концентрации газа с учетом пучка
def calculate_diffusion_flow_with_beam(vars):

    l, dl, D, inflow, n_beam, v_beam, n_gas_prev = vars
    # Сечения
    sigmas_beam = load_sigmas()
    sigma_neutr = np.sum(sigmas_beam[0][[0, 2, 3, 4, 5, 6]])
    sigma_proton = np.sum(sigmas_beam[1][[0, 1, 2, 3]])

    t_pass = l[-1]/((3*kB*T/M2)**0.5)
    t_max = 50*t_pass

    def diffusion(t, n):
        dndt = np.zeros_like(n)
        for i in range(1, len(dndt) - 1):
            electron_depletion = 0#[i]*ne[i]*sigmaV
            depletion = - n[i]*n_beam[0][i]*v_beam*sigma_neutr - n[i]*n_beam[1][i]*v_beam*sigma_proton
            d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
            dndt[i] = D*d2n_dx2 + depletion + electron_depletion

        depletion_0 =  - n[i]*n_beam[0][i]*v_beam*sigma_neutr - n[i]*n_beam[1][i]*v_beam*sigma_proton
        electron_depletion_0 = 0
        dndt[0] = 2*D*(n[1] - n[0])/dl**2 + 2*inflow/dl + depletion_0 + electron_depletion_0

        depletion_end = - n[i]*n_beam[0][i]*v_beam*sigma_neutr - n[i]*n_beam[1][i]*v_beam*sigma_proton
        electron_depletion_end = 0
        dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2 + depletion_end + electron_depletion_end

        return dndt
    
    n_IC = np.zeros(len(n_gas_prev))#n_gas_prev
    solution = solve_ivp(fun=diffusion,
                        t_span=(0, t_max),
                        y0=n_IC,
                        method='RK45')
        
    return solution

# ==================================================================================================
# Электроны
# Функция для загрузки сечени электронов из файла
def load_electron_sigmas():
    # На этих сетках будет интерполяция
    nrg = np.linspace(1e-3, 200, 5000)
    velocity = np.sqrt(2*nrg*qe/me)

    path = 'props/electrons_sigmas/'
    # Загружаем и интерполируем сечение на заданной сетке по энергии
    sigma_electron_ionization_molecular = np.loadtxt(path + 'Molecules_ionization_by_electron.txt')
    E = sigma_electron_ionization_molecular[:, 0]
    vnrg = np.sqrt(2*E*qe/me)
    sigma_molecular = sigma_electron_ionization_molecular[:, 1]
    sigma_molecular_interpolator = interp1d(E, sigma_molecular, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_molecular = sigma_molecular_interpolator(nrg)
    
    sngr_molecular_interpolator = interp1d(vnrg, sigma_molecular, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_vel_molecular = sngr_molecular_interpolator(velocity)

    del sngr_molecular_interpolator
    del sigma_molecular_interpolator

    sigma_electron_0 = sigma_electron_ionization_molecular
    sigma_electron_0_vel = sigma_electron_ionization_vel_molecular
    
    #####

    sigma_electron_ionization_atomar = np.loadtxt(path + 'Atoms_ionization_by_electron.txt')
    E = sigma_electron_ionization_atomar[:, 0]
    vnrg = np.sqrt(2*E*qe/me)
    sigma_atomar = sigma_electron_ionization_atomar[:, 1]
    sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

    sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

    del sngr_atomar_interpolator
    del sigma_atomar_interpolator

    sigma_electron_1 = sigma_electron_ionization_atomar
    sigma_electron_1_vel = sigma_electron_ionization_vel_atomar

    #######

    sigma_electron_ionization_atomar = np.loadtxt(path + 'Molecular_ion_dissociation_with_one_ion_and_ionization_second_atom_by_electron.txt')
    E = sigma_electron_ionization_atomar[:, 0]
    vnrg = np.sqrt(2*E*qe/me)
    sigma_atomar = sigma_electron_ionization_atomar[:, 1]
    sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

    sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

    del sngr_atomar_interpolator
    del sigma_atomar_interpolator

    sigma_electron_2 = sigma_electron_ionization_atomar
    sigma_electron_2_vel = sigma_electron_ionization_vel_atomar

    #######

    sigma_electron_ionization_atomar = np.loadtxt(path + 'Molecular_ion_dissociation_with_one_atom_and_second_ion_neutralization_by_electron.txt')
    E = sigma_electron_ionization_atomar[:, 0]
    vnrg = np.sqrt(2*E*qe/me)
    sigma_atomar = sigma_electron_ionization_atomar[:, 1]
    sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

    sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

    del sngr_atomar_interpolator
    del sigma_atomar_interpolator

    sigma_electron_3 = sigma_electron_ionization_atomar
    sigma_electron_3_vel = sigma_electron_ionization_vel_atomar

    # Это сечение очень странное, оно максимально при 0,1 эВ, пусть оно пока нулем побудет

    #######

    sigma_electron_ionization_atomar = np.loadtxt(path + 'Molecular_ion_dissociation_with_one_ion_and_one_atom_by_electron.txt')
    E = sigma_electron_ionization_atomar[:, 0]
    vnrg = np.sqrt(2*E*qe/me)
    sigma_atomar = sigma_electron_ionization_atomar[:, 1]
    sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

    sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
    sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

    del sngr_atomar_interpolator
    del sigma_atomar_interpolator

    sigma_electron_4 = sigma_electron_ionization_atomar
    sigma_electron_4_vel = sigma_electron_ionization_vel_atomar

    ## Расшифровка сечений
    # sigma_electron_0 Ионизация молекулы
    # sigma_electron_1 Ионизация атома
    # sigma_electron_2 Диссоциация молекулярного иона с рождением иона и ионизацией атома
    # sigma_electron_3 Диссоциация молекулярного иона с нейтрализацией иона и образованием атома
    # sigma_electron_4 Диссоциация молекулярного иона с образованием атома и иона

    return nrg, velocity, sigma_electron_0, sigma_electron_1, \
        sigma_electron_2,sigma_electron_3, sigma_electron_4, \
            sigma_electron_0_vel, sigma_electron_1_vel, sigma_electron_2_vel, \
                sigma_electron_3_vel, sigma_electron_4_vel

# Расчет усредненных sigmaV
def calc_sigmaV(Te, velocity, sigma_electron_0_vel, sigma_electron_1_vel, sigma_electron_2_vel, sigma_electron_3_vel, sigma_electron_4_vel):

    # распредление Максвелла
    maxvell = 4*np.pi*velocity**2*(me/(2*np.pi*kB*Te*11600))**(3/2)*np.exp(-(me*velocity**2)/(2*kB*Te*11600))

    # Усредненные значения
    sigma_e_V_0 = trapezoid(maxvell*sigma_electron_0_vel, velocity)/trapezoid(maxvell, velocity)
    sigma_e_V_1 = trapezoid(maxvell*sigma_electron_1_vel, velocity)/trapezoid(maxvell, velocity)
    sigma_e_V_2 = trapezoid(maxvell*sigma_electron_2_vel, velocity)/trapezoid(maxvell, velocity)
    sigma_e_V_4 = trapezoid(maxvell*sigma_electron_3_vel, velocity)/trapezoid(maxvell, velocity)
    sigma_e_V_3 = trapezoid(maxvell*sigma_electron_4_vel, velocity)/trapezoid(maxvell, velocity)

    return sigma_e_V_0, sigma_e_V_1, sigma_e_V_2, sigma_e_V_3, sigma_e_V_4
# ==================================================================================================
















