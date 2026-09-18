from matplotlib.pylab import rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from initiate_constants import *
import auxillary


## Cоздаем стартовые раcспределения координат
l = np.linspace(0, L, section_amount)
particles_start_position = np.zeros((len(l), N, 2))
for i in range(len(l)):
    a_middle, b_middle = auxillary.find_beam_profile(l[i])
    s_a, s_b = auxillary.distibute_particles(a1, b1, a_middle, b_middle, N)
    particles_start_position[i, :, 0] = s_a[:]
    particles_start_position[i, :, 1] = s_b[:]

## Создаем стартовые распределения скорости.
# Решать будем в 3д, поэтому будем задавать 3 проекции скорости. Задавать их будем по известному распределению по жнергии и по углу.
# Будем отдельно просчитывать в каждом сечении.
def moving(t, vars):
    # y[0] - это координата (x)
    # y[1] - это скорость (vx)
    # y[2] - это координата (y)
    # y[3] - это скорость (vy)
    # y[4] - это координата (z)
    # y[5] - это скорость (vz)
    #
    # ВАЖНО: функция должна быть "чистой" — только производные.
    # Обработка столкновений вынесена в события solve_ivp (см. ниже).
    # Если менять vars прямо здесь, адаптивный RK45 многократно вызывает
    # функцию вблизи стенки и частица начинает "дрожать" (ложные отражения).
    return [vars[1], 0.0, vars[3], 0.0, vars[5], 0.0]


# --- События solve_ivp: вылет и столкновения со стенками ---
# direction задаёт направление пересечения нуля, при котором событие срабатывает.

def leave_neutr(t, vars):
    return vars[0] - L
leave_neutr.terminal = True
leave_neutr.direction = 1

def hit_left(t, vars):
    return vars[2] + a/2
hit_left.terminal = True
hit_left.direction = -1

def hit_right(t, vars):
    return vars[2] - a/2
hit_right.terminal = True
hit_right.direction = 1

def hit_bottom(t, vars):
    return vars[4] + b/2
hit_bottom.terminal = True
hit_bottom.direction = -1

def hit_top(t, vars):
    return vars[4] - b/2
hit_top.terminal = True
hit_top.direction = 1

def hit_entrance(t, vars):
    return vars[0]
hit_entrance.terminal = True
hit_entrance.direction = -1


def trace_particle(vars0, t_max):
    """Интегрирует траекторию одной частицы, обрабатывая столкновения
    через события solve_ivp и перезапуск интегрирования с отражённой
    скоростью. Возвращает (t_exit, collisions, t_array, y_array)."""
    vars = np.array(vars0, dtype=float)
    t0 = 0.0
    collisions = 0
    t_hist = [0.0]
    y_hist = [vars.copy()]
    # Точки столкновений: (t, состояние, индекс стенки)
    collision_marks = []

    events = [leave_neutr, hit_left, hit_right, hit_bottom, hit_top, hit_entrance]

    while t0 < t_max:
        sol = solve_ivp(
            fun=moving,
            t_span=(t0, t_max),
            y0=vars,
            events=events,
            method='RK45',
            max_step=1e-7,
            rtol=1e-8, atol=1e-10)

        # Накапливаем траекторию
        for i in range(1, len(sol.t)):
            t_hist.append(sol.t[i])
            y_hist.append(sol.y[:, i].copy())

        # Вылет из нейтрализатора
        if len(sol.t_events[0]) > 0:
            return sol.t_events[0][0], collisions, np.array(t_hist), np.array(y_hist).T, collision_marks

        # Ищем самое раннее столкновение со стенкой
        earliest_t = None
        earliest_idx = None
        for idx in range(1, len(sol.t_events)):
            ev = sol.t_events[idx]
            if len(ev) > 0 and (earliest_t is None or ev[0] < earliest_t):
                earliest_t = ev[0]
                earliest_idx = idx

        if earliest_idx is None:
            # Ничего не произошло — таймаут
            return t_max, collisions, np.array(t_hist), np.array(y_hist).T, collision_marks

        # Состояние в момент столкновения
        vars = sol.y_events[earliest_idx][0].copy()
        t0 = earliest_t
        collisions += 1
        collision_marks.append((earliest_t, vars.copy(), earliest_idx))

        # Диффузное отражение от всех стенок и входного сечения
        new_v = auxillary.diffuse_reflection(vars, earliest_idx)
        vars[1] = new_v[0]
        vars[3] = new_v[1]
        vars[5] = new_v[2]

    return t_max, collisions, np.array(t_hist), np.array(y_hist).T, collision_marks

# k - индексация по длине
# n - индекация по номеру частицы

mean_energy, mean_angle = auxillary.section_treater()
vx0, v_transverse = auxillary.calculate_velocity(mean_energy, mean_angle)

time_alive_overall = []
collisions_overall = []
# Данные по каждой частице каждого сечения (для гистограмм и сохранения)
time_alive_per_section = []
collisions_per_section = []
# Потом пробегаем по всем сечениям
for k in range(len(l)):
    # В начале каждого сечения генерируем скорости (координаты тоже хорошо бы генерировать в начале каждого сечения)
    
    vy0, vz0 = auxillary.distribute_velocity_projections(N, v_transverse)
    # Определяем среднее время жизни частицы в нейтрализаторе, 
    tau_analitic_mean = (L - l[k])/vx0.mean()
    # Убеждаемся, что время расчета не ноль
    if k == len(l) - 2:
        tau_analitic_mean_cache = tau_analitic_mean
    if tau_analitic_mean == 0:
        tau_analitic_mean = tau_analitic_mean_cache

    time_ailve_section = []
    collisions_section = []
    # Сначала пробегаем по всем частицам в сечении
    for n in range(N):

        vars0 = [l[k], vx0[n], particles_start_position[k, n, 0], vy0[n], particles_start_position[k, n, 1], vz0[n]]
        # Умножаем на 20 для достоверности
        t_max = 20*tau_analitic_mean
        t_leave, collisions, t_plot, y_plot, collision_marks = trace_particle(vars0, t_max)
        
        time_ailve_section.append(t_leave)
        collisions_section.append(collisions)
        
        
        # # Вывод траектории частицы (проекции x-y и x-z) с отметками столкновений
        # auxillary.plot_trajectory(y_plot, collision_marks, collisions)
    
    ## Отдально сохраняем для каждого сечения все данные
    savename = f'section_data.txt'
    savepath = './output/'
    savefilepath = savepath + savename
    
    with open(savefilepath, 'w', encoding='utf-8') as f:
        f.write(f'# section {k}\n')
        for val in time_ailve_section:
            f.write(f'time alive {val}\n')
    exit()

    # Сохраняем данные по частицам этого сечения (для гистограмм и файла)
    time_alive_per_section.append(time_ailve_section)
    collisions_per_section.append(collisions_section)

    # time_alive_overall.append(np.mean(time_ailve_section))
    # collisions_overall.append(np.mean(collisions_section))

# Сохраняем все данные по частицам в txt-файл с пояснениями
auxillary.save_section_data(
    'section_data.txt',
    l,
    time_alive_per_section,
    collisions_per_section)

    

