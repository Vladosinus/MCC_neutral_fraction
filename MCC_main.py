from matplotlib.pylab import rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from initiate_constants import *
import auxillary

# Путь для сохранения файла
savename = f'mean_vals.txt'
savepath = './output/'
savefilepath = savepath + savename

# Путь к файлу с данными по частицам каждого сечения, ВЫВОД
section_data_path = 'section_data.txt'

# Базовое зерно генератора случайных чисел для трассировки частиц.
# Каждая частица получает своё зерно (seed_base + k*N + n), поэтому
# результат воспроизводим и не зависит от порядка выполнения задач
# при распараллеливании.
seed_base = 42

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
hit_entrance.direction = -2


def trace_particle(vars0, t_max, collect_trajectory=True):
    """Интегрирует траекторию одной частицы, обрабатывая столкновения
    через события solve_ivp и перезапуск интегрирования с отражённой
    скоростью. Возвращает (t_exit, collisions, t_array, y_array).

    collect_trajectory=False — не копить координаты траектории (нужны
    только время вылета и число столкновений). Это заметно экономит
    память и время при массовом расчёте в параллельных процессах."""
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

        # Накапливаем траекторию (только если она нужна)
        if collect_trajectory:
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

def _trace_particle_task(task):
    """Задача для пула процессов: трассировка одной частицы.

    Принимает (vars0, t_max, seed) и возвращает только (t_leave, collisions),
    чтобы не пересылать между процессами тяжёлые массивы траектории.
    Собственное зерно генератора делает вклад частицы воспроизводимым и
    не зависящим от порядка выполнения задач."""
    vars0, t_max, seed = task
    np.random.seed(seed)
    t_leave, collisions, _, _, _ = trace_particle(vars0, t_max, collect_trajectory=False)
    return t_leave, collisions


def trace_particles_batch(tasks, executor=None, workers=1):
    """Трассирует набор частиц.

    tasks    - список кортежей (vars0, t_max, seed)
    executor - ProcessPoolExecutor для параллельного расчёта; если None,
               расчёт идёт последовательно в текущем процессе
    workers  - число рабочих процессов (нужно для оценки chunksize)

    Возвращает список (t_leave, collisions)."""
    if executor is None:
        return [_trace_particle_task(task) for task in tasks]
    # chunksize снижает накладные расходы на передачу задач между процессами
    chunksize = max(1, len(tasks)//(workers*4))
    return list(executor.map(_trace_particle_task, tasks, chunksize=chunksize))


def _write_section_stats(k, l_k, time_ailve_section, collisions_section):
    """Дописывает в mean_vals.txt времена жизни и числа столкновений всех
    частиц сечения, а также их средние значения."""
    mode = 'a' if os.path.exists(savefilepath) else 'w'
    with open(savefilepath, mode, encoding='utf-8') as f:
        f.write(f'# section {k}, l = {l_k} м\n')
        f.write(f"{'time alive, seconds':>14} {'collisions':>12}\n")
        for v in range(len(time_ailve_section)):
            f.write(f'{time_ailve_section[v]:>14.3e} {collisions_section[v]:>12.1f}\n')
        f.write(f"{'mean time alive, seconds':>20} {'mean collisions':>20}\n")
        f.write(f'{np.mean(time_ailve_section):>20.3e} {np.mean(collisions_section):>20}\n')
        f.write(f'\n')
    print(f'Сечение номер {k}')


def _start_positions_task(l_i):
    """Задача для пула процессов: стартовые координаты частиц в сечении.

    Функции auxillary.find_beam_profile и auxillary.distibute_particles
    детерминированы (фиксированное зерно внутри), поэтому результат не
    зависит от того, в каком процессе задача выполнена."""
    a_middle, b_middle = auxillary.find_beam_profile(l_i)
    s_a, s_b = auxillary.distibute_particles(a1, b1, a_middle, b_middle, N)
    return s_a, s_b


def main():
    # Удаляем файл, если существует, чтобы всякого не случилось
    if os.path.exists(savefilepath):
        # Текущая папка
        base_dir = Path(__file__).resolve().parent
        # Откуда и куда (пути берём из savefilepath, чтобы настройки
        # сохранения и переноса файла всегда совпадали)
        src = Path(savefilepath)
        dst = base_dir / src.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        # os.remove(savefilepath)

    # Пул процессов для распараллеливания независимых расчётов
    executor = None
    workers = n_workers if n_workers else os.cpu_count()
    if use_multiprocessing and workers and workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        print(f'Распараллеливание включено: {workers} процессов')

    try:
        ## Cоздаем стартовые раcспределения координат
        l = np.linspace(0, L, section_amount)
        particles_start_position = np.zeros((len(l), N, 2))
        # Сечения независимы друг от друга, поэтому стартовые распределения
        # координат считаем параллельно
        if executor is None:
            positions = [_start_positions_task(l_i) for l_i in l]
        else:
            positions = list(executor.map(_start_positions_task, l))
        for i in range(len(l)):
            particles_start_position[i, :, 0] = positions[i][0][:]
            particles_start_position[i, :, 1] = positions[i][1][:]

        # Находим средние значения угла и энергии из распределения и назначаем
        # стартовые скорости для частиц
        mean_energy, mean_angle = auxillary.section_treater()
        vx0, v_transverse = auxillary.calculate_velocity(mean_energy, mean_angle)

        time_alive_overall = []
        collisions_overall = []
        # Данные по каждой частице каждого сечения (для гистограмм и сохранения)
        time_alive_per_section = []
        collisions_per_section = []

        # Потом пробегаем по всем сечениям
        for k in range(len(l)):
            # В начале каждого сечения генерируем скорости (координаты тоже
            # хорошо бы генерировать в начале каждого сечения)
            vy0, vz0 = auxillary.distribute_velocity_projections(N, v_transverse)
            # Определяем среднее время жизни частицы в нейтрализаторе
            tau_analitic_mean = (L - l[k])/vx0.mean()
            # Убеждаемся, что время расчета не ноль
            if k == len(l) - 2:
                tau_analitic_mean_cache = tau_analitic_mean
            if tau_analitic_mean == 0:
                tau_analitic_mean = tau_analitic_mean_cache

            # Умножаем на 20 для достоверности
            t_max = 200*tau_analitic_mean
            # Готовим независимые задачи по частицам сечения: каждая задача
            # несёт стартовые условия, время расчёта и собственное зерно ГСЧ.
            # Такие задачи не связаны друг с другом, поэтому их можно считать
            # параллельно.
            tasks = []
            for n in range(N):
                vars0 = [l[k], vx0[n], particles_start_position[k, n, 0],
                         vy0[n], particles_start_position[k, n, 1], vz0[n]]
                tasks.append((vars0, t_max, seed_base + k*N + n))

            # Считаем частицы сечения (параллельно, если есть пул процессов)
            results = trace_particles_batch(tasks, executor, workers)

            # Времена жизни и количества соударений в пределах сечения
            time_ailve_section = [res[0] for res in results]
            collisions_section = [res[1] for res in results]

            # # Вывод траектории частицы (проекции x-y и x-z) с отметками столкновений
            # auxillary.plot_trajectory(y_plot, collision_marks, collisions)

            ## Отдельно сохраняем для каждого сечения все данные
            _write_section_stats(k, l[k], time_ailve_section, collisions_section)
            # exit()

            # Сохраняем данные по частицам этого сечения (для гистограмм и файла),
            # внутри переменной набор (количество равно числу сечений) массивов,
            # в котором лежат параметры каждой частицы внутри сечения
            time_alive_per_section.append(time_ailve_section)
            collisions_per_section.append(collisions_section)

            # time_alive_overall.append(np.mean(time_ailve_section))
            # collisions_overall.append(np.mean(collisions_section))
    finally:
        # Корректно останавливаем пул процессов в любом случае
        if executor is not None:
            executor.shutdown()

    # Сохраняем все данные по частицам в txt-файл с пояснениями
    # auxillary.save_section_data(
    #     section_data_path,
    #     l,
    #     time_alive_per_section,
    #     collisions_per_section)


if __name__ == '__main__':
    main()



