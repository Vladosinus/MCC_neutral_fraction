
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
    s_a, s_b = auxillary.distibute_particles(a1, b1, N)
    return s_a, s_b
