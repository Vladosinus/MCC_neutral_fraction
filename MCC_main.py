from matplotlib.pylab import rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from initiate_constants import *
import auxillary


## Cоздаем стартовые раcспределения координат
l = np.linspace(0, L, 10)
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

    ## Обработка столкновений

# Левая стенка
    if vars[2] <= -a/2:
        # Находим полную скорость
        speed = (vars[1]**2 + vars[3]**2 + vars[5]**2)**0.5
        # Сначала смотрим по нормальному углу, куда попало
        normal = np.deg2rad(np.random.rand()*180)
        vars[3] = -speed*np.sin(normal)
        v_longitudinal = speed*np.cos(normal)
        # Азимутальный
        azimuth = np.deg2rad(np.random.rand()*360)
        vars[1] = v_longitudinal*np.cos(azimuth)
        vars[5] = v_longitudinal*np.sin(azimuth)

    # Правая стенка
    if vars[2] >= a/2:
        vars[3] = -vars[3]

    # Нижняя стенка
    if vars[4] <= -b/2:
        vars[5] = -vars[5]

    # Верхняя стенка
    if vars[4] >= b/2:
            vars[5] = -vars[5]

    if vars[0] <= 0:
        vars[1] = -vars[1]
    Сделать провека по предыдущему состоянию, вдруг частица колеблется около стенки
    # Возвращаем массив производных [dx/dt, dv/dt]
    dx_dt = vars[1]
    dvx_dt1 = 0
    dy_dt = vars[3]
    dvy_dt1 = 0
    dz_dt = vars[5]
    dvz_dt1 = 0

    return [dx_dt, dvx_dt1, dy_dt, dvy_dt1, dz_dt, dvz_dt1]

def leave_neutr(t, vars):
    return vars[0] - L
leave_neutr.terminal = True
leave_neutr.direction = 1

# k - индексация по длине
# n - индекация по номеру частицы

mean_energy, mean_angle = auxillary.section_treater()
vx0, v_transverse = auxillary.calculate_velocity(mean_energy, mean_angle)

time_alive_overall = []
collisions_overall = []
# Потом пробегаем по всем сечениям
for k in range(len(l)):
    # В начале каждого сечения генерируем скорости (координаты тоже хорошо бы генерировать в начале каждого сечения)
    
    vy0, vz0 = auxillary.distribute_velocity_projections(N, v_transverse)
    # Определяем среднее время жизни частицы в нейтрализаторе, умножаем на 3 для достоверности (из среднего угла и чисто геометрических соображений)
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
        
        t_span = (0, 20*tau_analitic_mean)
        t_eval = np.linspace(0, 20*tau_analitic_mean, 100)
        solution = solve_ivp(
            fun = moving, 
            t_span = t_span, 
            y0 = vars0, 
            events = leave_neutr, 
            dense_output = True,
            method = 'RK45',
            max_step = 1e-5,
            rtol = 1e-8, atol = 1e-10)
        
        t_hit = solution.t_events[0][0]
        t_plot = np.linspace(0, t_hit, 100)
        y_plot = solution.sol(t_plot)

        collisions = auxillary.collision_counter(t_plot, y_plot)
                
        time_ailve_section.append(t_hit)
        collisions_section.append(collisions)
        
        exit()
        
    time_alive_overall.append(np.mean(time_ailve_section))
    collisions_overall.append(np.mean(collisions_section))
    print(f'Среднее время жизни частиц из сечения: {time_alive_overall[0]*1e6:.1f}, мкс')
    print(f'Среднее количество ударов о стенку частиц из сечения: {collisions_overall[0]:.1f}, шт')
    print(f'Средняя энергия вторичных частиц по распределению: {mean_energy:.1f}, эВ')
    print(f'Средний угол вылета вторичных частиц по распределению: {mean_angle:.1f}, град')
    exit()
    

