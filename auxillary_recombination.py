# from concurrent.futures.process import _threads_wakeups
import multiprocessing as mp
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


from initiate_constants import *

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

    # print(len(sigmas[:, 0]))
    # print(len(secondary_energy))
    # exit()

    ## Средний угол
    only_angle_distibution = np.zeros(len(sigmas[0, :]))
    for i in range(len(sigmas[0, :])):
        only_angle_distibution[i] = np.trapezoid(sigmas[:, i], secondary_energy)

    #  Находим средние значения энергии и угла
    mean_energy = np.trapezoid(only_energy_distibution*secondary_energy, secondary_energy)/np.trapezoid(only_energy_distibution, secondary_energy)
    mean_angle = np.trapezoid(only_angle_distibution*thetas, thetas)/np.trapezoid(only_angle_distibution, thetas)

    return mean_energy, mean_angle

## Находим проекции скорости
def calculate_velocity(mean_energy, mean_angle):
            
    speed = (2*mean_energy*qe/M)**0.5
    
    vx0 = speed*np.abs(np.cos(np.deg2rad(mean_angle)))
    v_transverse = speed*np.abs(np.sin(np.deg2rad(mean_angle)))
    return vx0, v_transverse

## Распределяем скорости по азимутальному углу
# ==========================================================================
# Генератор заданий
def _velocity_projections(args):
    i, j, N, v_transverse = args
    vy0 = np.zeros(N)
    vz0 = np.zeros(N)
    
    for n in range(N):
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
    
    return i, j, vy0, vz0
# Органзатор
def distribute_velocity_projections(la, lb, v_transverse):
    print('Назначение исходных скоростей...')
    # Число потоков
    n_workers = os.cpu_count()
    # Пул задач
    tasks = []
    for i in range(len(la)):
        for j in range(len(lb)):
            tasks.append((i, j, N, v_transverse))


    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_velocity_projections, tasks))
    
    vel_y = np.zeros((len(la), len(lb), N))
    vel_z = np.zeros((len(la), len(lb), N))
    # распаковываем результат
    for i, j, vy_slice, vz_slice in results:
        vel_y[i, j, :] = vy_slice
        vel_z[i, j, :] = vz_slice

    return vel_y, vel_z

# ==========================================================================
# Скорости при диффузном отражении
def diffuse_reflection(vars, wall_idx):
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
    speed_old = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
    nrg = (M*speed_old**2)/2
    deltaE = nrg - 300/11604
    loss = 0.102*deltaE
    nrg_new = nrg - loss
    speed = (2*nrg_new/M)**0.5
    
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
## Трассировка частиц, правая часть и ивенты
# Правая часть
def moving(t, part_prop):
    return [part_prop[1], 0, part_prop[3], 0, part_prop[5], 0]

# Ивенты
# ------------------------------------
def leave_neutr(t, part_prop):
    return part_prop[0] - L
leave_neutr.terminal = True
leave_neutr.direction = 1

def hit_left(t, part_prop):
    return part_prop[2] + a/2
hit_left.terminal = True
hit_left.direction = -1

def hit_right(t, part_prop):
    return part_prop[2] - a/2
hit_right.terminal = True
hit_right.direction = 1

def hit_bottom(t, part_prop):
    return part_prop[4] + b/2
hit_bottom.terminal = True
hit_bottom.direction = -1

def hit_top(t, part_prop):
    return part_prop[4] - b/2
hit_top.terminal = True
hit_top.direction = 1

def hit_entrance(t, part_prop):
    return part_prop[0]
hit_entrance.terminal = True
hit_entrance.direction = -1
# ------------------------------------
# Генератор заданий
def _trace_atoms_pool(args):

    i, j, n, N, x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current, t_max, recombination_probability = args
    # Начальные условия
    vars0 = [x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current]
    
    # Ивенты
    events = [leave_neutr, hit_left, hit_right, hit_bottom, hit_top, hit_entrance]
    
    # Время, по которому расчет
    t0 = 0
    # Счетчик столкновений
    collisions = 0
    # Координаты и прочее для столкновений
    collision_marks = []
    # Заготовки под накопление траектории
    t_history = [0]    
    prop_history = [np.array(vars0).copy()]
    recombined = False
    
    # Так как используем ивенты, будем много перезапускать
    while t0 < t_max:
        # Считаем движение, так как есть ивенты счет остановится при соударении со стенкой
        
        solution = solve_ivp(
                            fun = moving,
                            t_span = (t0, t_max),
                            y0 = vars0,
                            method = 'RK45',
                            events = events,
                            max_step = t_max/5,
                            rtol=1e-8, atol=1e-10)

        # Накапливаетс ятраектория
        for i_history in range(1, len(solution.t)):
            t_history.append(solution.t[i_history])
            prop_history.append(solution.y[:, i_history].copy())            

        # Если просто вылет из нейтрализатора
        if len(solution.t_events[0]) > 0:
            return recombined, i, j, n, np.array(t_history), prop_history, collisions, collision_marks

        # Первое столкновение со стенкой
        # Время, когда случилось соударение
        earliest_t = None
        # Индекс стенки, соответствцует индексу ивента в списке ивентов
        earliest_idx = None
        # Проходим по всме ивентам и ищем, какой из них сработал
        for idx in range(1, len(solution.t_events)):
            ev = solution.t_events[idx]
            if len(ev) > 0 and (earliest_t is None or ev[0] < earliest_t):
                earliest_t = ev[0]
                earliest_idx = idx

        if earliest_idx is None:
            # Ничего не произошло — таймаут? за все выделенное время остался в нейтрализаторе и не ударился, а остался в воздухе висеть
            return recombined, i, j, n, np.array(t_history), prop_history, collisions, collision_marks

        # Рекомбинация, если есть, заканчиваем, если нет, то считаем дальше
        if np.random.rand() <= recombination_probability:
            # Заканичваем расчет, ставим флаг
            recombined = True
            return recombined, i, j, n, np.array(t_history), prop_history, collisions, collision_marks
        else:
            # Если рекомбинации не было, что продолжаем расчет
            # Переписываем координаты, где произошло столкновение
            vars0 = solution.y_events[earliest_idx][0]
            t0 = earliest_t
            collisions += 1
            collision_marks.append((earliest_t, vars0, earliest_idx))

            # Переписываем начальные скорости по диффузному отражению и обрабатываем возможные события
            new_v = diffuse_reflection(vars0, earliest_idx)
            vars0[1] = new_v[0]
            vars0[3] = new_v[1]
            vars0[5] = new_v[2]
        
    return recombined, i, j, n, np.array(t_history), prop_history, collisions, collision_marks
    
# Организатор
def trace_atoms(L, lk, a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0, recombination_probability):
    print(f'Начата трассировка сечения {lk} м ...')
    n_workers = os.cpu_count()
    t_max = L/vx0*5
    tasks = []
    # Пул задач
    for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                for n in range(N):
                    tasks.append((i, j, n, N, lk, vx0, y_cell[i, j, n], vy0[i, j, n], z_cell[i, j, n], vz0[i, j, n], t_max, recombination_probability))
    chunksize = max(1, len(tasks)//(n_workers*4))
    # Сам расчет
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_trace_atoms_pool, tasks, chunksize=chunksize))

    # Подготовка переменных под распаковку результата
    is_recombined = np.zeros((len(a_grid), len(b_grid), N))
    time = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    particles_trajectoies = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    collisions_count = np.zeros((len(a_grid), len(b_grid), N))
    collision_points = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    

    # Распаковка результата
    for recombined, i, j, n, t_history, prop_history, collisions, collision_marks in results:
        time[i, j, n] = np.array(t_history)
        particles_trajectoies[i, j, n] = np.array(prop_history)
        collisions_count[i, j, n] = collisions
        collision_points[i, j, n] = collision_marks
        is_recombined[i, j, n] = int(recombined)
    
    return time, particles_trajectoies, collisions_count, collision_points, is_recombined

# Правая часть
def moving(t, part_prop):
    return [part_prop[1], 0, part_prop[3], 0, part_prop[5], 0]

# ==========================================================================
# Расчет движения молекул с произвольными местами старта

























