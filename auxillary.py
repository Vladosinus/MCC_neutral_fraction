import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


from initiate_constants import *

def find_beam_profile(l):
    a2 = 2*l*np.tan(alpha) + a1
    b2 = 2*l*np.tan(beta) + b1
    return a2, b2

def distibute_particles(a1, b1, a_middle, b_middle, N):
    stretching_koef_a = a_middle/a1
    
    # Короткая сторона
    x_a = np.linspace(-a1/2, a1/2, 100)
    temp_a = np.loadtxt(filepath_short)
    temp_a[0] = -a1/2
    temp_a[-1] = a1/2
    density_a = interpolate.interp1d(temp_a[:, 0], temp_a[:, 1], kind = 'linear', bounds_error = False, fill_value = 0)
    y_a = density_a(x_a)
    y_a = y_a / np.trapezoid(y_a, x_a)
    x_a = x_a*stretching_koef_a

    cdf = np.zeros_like(x_a)
    for i in range(1, len(x_a)):
        cdf[i] = cdf[i-1] + np.trapezoid(y_a[i-1:i+1], x_a[i-1:i+1])
    cdf = cdf / cdf[-1]
    
    cdf_inverse = interpolate.interp1d(cdf, x_a, kind='linear', bounds_error=False, fill_value='extrapolate')
    rng = np.random.default_rng(seed=42)
    u_a = rng.random(N)
    s_a = cdf_inverse(u_a)


    # Длинная сторона
    stretching_koef_b = b_middle/b1
    x_b = np.linspace(-b1/2, b1/2, 100)
    temp_b = np.loadtxt(filepath_long)
    temp_b[0] = -b1/2
    temp_b[-1] = b1/2
    density_b = interpolate.interp1d(temp_b[:, 0], temp_b[:, 1], kind = 'linear', bounds_error = False, fill_value = 0)
    y_b = density_b(x_b)
    y_b = y_b / np.trapezoid(y_b, x_b)
    x_b = x_b*stretching_koef_b

    cdf = np.zeros_like(x_b)
    for i in range(1, len(x_b)):
        cdf[i] = cdf[i-1] + np.trapezoid(y_b[i-1:i+1], x_b[i-1:i+1])
    cdf = cdf / cdf[-1]
    from scipy.interpolate import interp1d
    cdf_inverse = interp1d(cdf, x_b, kind='linear', bounds_error=False, fill_value='extrapolate')
    rng = np.random.default_rng(seed=42)
    u_b = rng.random(N)
    s_b = cdf_inverse(u_b)

    return s_a, s_b

def calculate_velocity(mean_energy, mean_angle):
            
    speed = (2*mean_energy*qe/M)**0.5
    
    ## Назначаем скорости
    vx0 = np.zeros(N)
    
    vx0[:] = speed*np.abs(np.cos(np.deg2rad(mean_angle)))
    v_transverse = speed*np.abs(np.sin(np.deg2rad(mean_angle)))
    return vx0, v_transverse

def distribute_velocity_projections(N, v_transverse):

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

    
    return vy0, vz0

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
    theta10_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
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
    theta50_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
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
    theta90_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
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

def plot_trajectory(y_plot, collision_marks, collisions):
    """Строит две проекции траектории частицы (x-y и x-z) с отметками
    столкновений, разделёнными по типу стенки.

    Параметры:
        y_plot         - массив решения формы (6, M): строки [x, vx, y, vy, z, vz]
        collision_marks - список кортежей (t, состояние, индекс_стенки), где
                          индекс: 1 - левая (y=-a/2), 2 - правая (y=+a/2),
                                  3 - нижняя (z=-b/2), 4 - верхняя (z=+b/2),
                                  5 - вход (x=0)
        collisions     - общее число столкновений (для заголовка)
    """
    # Разделяем точки столкновений по типу стенки
    vert_marks = [(m[1][0], m[1][2]) for m in collision_marks if m[2] in (1, 2)]
    horiz_marks = [(m[1][0], m[1][4]) for m in collision_marks if m[2] in (3, 4)]

    # Проекция x-y
    plt.figure()
    plt.plot(y_plot[0], y_plot[2], '-', linewidth=1, label='траектория')
    plt.axhline(-a/2, color='k', linestyle='--', linewidth=0.8)
    plt.axhline(a/2, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(L, color='k', linestyle='--', linewidth=0.8)
    if vert_marks:
        vx_m, vy_m = zip(*vert_marks)
        plt.plot(vx_m, vy_m, 'ms', markersize=5, label='удар о вертикальную стенку')
    # Удары о горизонтальные стенки (z = ±b/2) — в проекции x-y видны как
    # изломы, так как при ударе меняется и vy
    if horiz_marks:
        hx_m, hy_m = zip(*[(m[1][0], m[1][2]) for m in collision_marks if m[2] in (3, 4)])
        plt.plot(hx_m, hy_m, 'c^', markersize=5, label='удар о горизонтальную стенку')
    plt.plot(y_plot[0, 0], y_plot[2, 0], 'go', label='старт')
    plt.plot(y_plot[0, -1], y_plot[2, -1], 'rx', label='финиш')
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title(f'Траектория частицы (проекция x-y), столкновений: {collisions}')
    plt.legend()
    plt.grid(True)

    # Проекция x-z
    plt.figure()
    plt.plot(y_plot[0], y_plot[4], '-', linewidth=1, label='траектория')
    plt.axhline(-b/2, color='k', linestyle='--', linewidth=0.8)
    plt.axhline(b/2, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(L, color='k', linestyle='--', linewidth=0.8)
    # Удары о вертикальные стенки (y = ±a/2) дают изломы в x-z, так как
    # при ударе меняется и vz
    if vert_marks:
        vx_m, vz_m = zip(*[(m[1][0], m[1][4]) for m in collision_marks if m[2] in (1, 2)])
        plt.plot(vx_m, vz_m, 'ms', markersize=5, label='удар о вертикальную стенку')
    if horiz_marks:
        hx_m, hz_m = zip(*horiz_marks)
        plt.plot(hx_m, hz_m, 'c^', markersize=5, label='удар о горизонтальную стенку')
    plt.plot(y_plot[0, 0], y_plot[4, 0], 'go', label='старт')
    plt.plot(y_plot[0, -1], y_plot[4, -1], 'rx', label='финиш')
    plt.xlabel('x, м')
    plt.ylabel('z, м')
    plt.title(f'Траектория частицы (проекция x-z), столкновений: {collisions}')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_section_data(filepath, l, time_alive_per_section, collisions_per_section):
    """Сохраняет времена жизни и количества столкновений каждой частицы
    из каждого сечения в текстовый файл с пояснениями.

    Параметры:
        filepath               - путь к txt-файлу
        l                      - массив координат сечений (длина = число сечений)
        time_alive_per_section - список списков: для каждого сечения список
                                 времён жизни частиц (сек)
        collisions_per_section - список списков: для каждого сечения список
                                 количеств столкновений частиц (шт)

    Формат файла: сначала идёт блок пояснений, затем для каждого сечения
    отдельный блок с координатой, числом частиц и таблицей
    "номер частицы | время жизни, с | число столкновений".
    """
    n_sections = len(l)
    with open(filepath, 'w', encoding='utf-8') as f:
        # --- Блок пояснений ---
        f.write("=" * 70 + "\n")
        f.write("ДАННЫЕ МОДЕЛИРОВАНИЯ ДВИЖЕНИЯ ЧАСТИЦ В НЕЙТРАЛИЗАТОРЕ\n")
        f.write("=" * 70 + "\n")
        f.write("\n")
        f.write("ОПИСАНИЕ СТРУКТУРЫ ФАЙЛА:\n")
        f.write("  - Частицы стартуют из сечений, расположенных вдоль оси x.\n")
        f.write("  - Координаты сечений задаются массивом l (в метрах).\n")
        f.write("  - Из каждого сечения запускается N модельных частиц.\n")
        f.write("  - Для каждой частицы фиксируются:\n")
        f.write("      * время жизни в нейтрализаторе (сек) - время от старта\n")
        f.write("        до вылета через сечение x = L;\n")
        f.write("      * число столкновений со стенками (шт).\n")
        f.write("\n")
        f.write("СТРУКТУРА ДАННЫХ:\n")
        f.write("  Для каждого сечения идёт блок вида:\n")
        f.write("    SECTION <номер> | x = <координата>, м | N = <число частиц>\n")
        f.write("    particle_index  time_alive_s  collisions\n")
        f.write("    <номер>         <время, с>    <столкновений>\n")
        f.write("    ...\n")
        f.write("\n")
        f.write(f"Число сечений: {n_sections}\n")
        f.write(f"Координаты сечений l, м: {', '.join(f'{x:.6g}' for x in l)}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("ДАННЫЕ ПО СЕЧЕНИЯМ\n")
        f.write("=" * 70 + "\n")

        # --- Данные по каждому сечению ---
        for k in range(n_sections):
            times = time_alive_per_section[k]
            colls = collisions_per_section[k]
            f.write("\n")
            f.write(f"# SECTION {k} | x = {l[k]:.6g}, м | N = {len(times)}\n")
            f.write(f"# {'particle_index':>14} {'time_alive_s':>14} {'collisions':>12}\n")
            for n in range(len(times)):
                f.write(f"  {n:>14d} {times[n]:>14.6e} {colls[n]:>12d}\n")

        # --- Итоговая сводка ---
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("СВОДКА (средние значения по сечениям)\n")
        f.write("=" * 70 + "\n")
        f.write(f"# {'section':>8} {'x, м':>12} {'mean_time_s':>16} "
                f"{'mean_collisions':>18}\n")
        for k in range(n_sections):
            mean_t = np.mean(time_alive_per_section[k])
            mean_c = np.mean(collisions_per_section[k])
            f.write(f"  {k:>8d} {l[k]:>12.6g} {mean_t:>16.6e} {mean_c:>18.4f}\n")

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
    speed = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)

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

