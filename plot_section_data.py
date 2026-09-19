"""Чтение результатов моделирования из mean_vals.txt и построение графиков.

Скрипт:
  1. Читает файл mean_vals.txt (создаётся MCC_main.py).
  2. Строит гистограммы времени жизни и числа столкновений для выбранных
     сечений (два окна на одном графике).
  3. Строит два графика зависимостей от координаты l: среднее время жизни
     в сечении и среднее число столкновений со стенкой.

Формат mean_vals.txt — повторяющиеся блоки по сечениям:
    # section <k>, l = <координата> м
    time alive, seconds   collisions
    <время частицы, с>    <столкновений частицы, шт>
    ...
    mean time alive, seconds      mean collisions
    <среднее время, с>            <среднее столкновений, шт>
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


from initiate_constants import *


# Путь к файлу результатов. None — искать автоматически среди CANDIDATE_PATHS.
FILEPATH = None

# Где может лежать mean_vals.txt. В 'output/' — результат последнего прогона,
# в корне — перенесённый результат предыдущего прогона (MCC_main.py при старте
# переносит старый output/mean_vals.txt в корень).
CANDIDATE_PATHS = ['output/mean_vals.txt', 'mean_vals.txt']


def find_data_file(candidates=None):
    """Возвращает путь к самому свежему существующему файлу результатов
    (None, если ни один из CANDIDATE_PATHS не найден)."""
    if candidates is None:
        candidates = CANDIDATE_PATHS
    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)

# Номера сечений, для которых строятся гистограммы (индексы из файла)
SELECTED_SECTIONS = [0]

# Число бинов для гистограмм
BINS = 30


def read_section_data(filepath):
    """Читает mean_vals.txt с данными по сечениям.

    Возвращает:
        sections - список словарей вида
                   {'index': k, 'x': x, 'times': np.array,
                    'collisions': np.array,
                    'mean_time': float, 'mean_collisions': float}

    Формат файла (повторяется для каждого сечения):
        # section <k>, l = <координата> м
        time alive, seconds   collisions
        <время частицы, с>    <столкновений частицы, шт>
        ...
        mean time alive, seconds      mean collisions
        <среднее время, с>            <среднее столкновений, шт>
    """
    sections = []
    current = None
    # После заголовка "mean time alive..." идёт строка со средними значениями,
    # её нужно сохранить отдельно, а не как ещё одну частицу.
    expect_mean = False

    section_re = re.compile(r'#\s*section\s+(\d+)\s*,\s*l\s*=\s*([-\d.eE+]+)')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            m = section_re.match(stripped)
            if m:
                # Начался новый блок сечения
                if current is not None:
                    sections.append(current)
                current = {
                    'index': int(m.group(1)),
                    'x': float(m.group(2)),
                    'times': [],
                    'collisions': [],
                    'mean_time': None,
                    'mean_collisions': None,
                }
                expect_mean = False
                continue

            # Прочие строки-комментарии пропускаем
            if stripped.startswith('#'):
                continue

            if current is None:
                continue

            # Заголовок "mean time alive, seconds   mean collisions":
            # следующая числовая строка — средние значения по сечению
            if stripped.lower().startswith('mean'):
                expect_mean = True
                continue

            parts = stripped.split()
            try:
                t = float(parts[0])
                c = float(parts[1])
            except (ValueError, IndexError):
                # Строка-заголовок "time alive, seconds   collisions" и т.п.
                continue

            if expect_mean:
                current['mean_time'] = t
                current['mean_collisions'] = c
                expect_mean = False
                continue

            current['times'].append(t)
            current['collisions'].append(c)

    if current is not None:
        sections.append(current)

    # Преобразуем списки в массивы
    for s in sections:
        s['times'] = np.array(s['times'])
        s['collisions'] = np.array(s['collisions'])

    return sections


def main():
    filepath = FILEPATH if FILEPATH else find_data_file()
    if filepath is None:
        print('Не найден файл результатов. Проверьте CANDIDATE_PATHS: '
              + ', '.join(CANDIDATE_PATHS))
        return

    sections = read_section_data(filepath)
    if not sections:
        print(f'Не удалось прочитать данные из {filepath}')
        return

    print(f'Чтение данных из {filepath}')

    # Индексация по номеру сечения
    by_index = {s['index']: s for s in sections}

    # Массивы средних значений по всем сечениям (для графиков от l)
    l_all = np.array([s['x'] for s in sections])
    mean_time_all = np.array([np.mean(s['times']) for s in sections])
    mean_coll_all = np.array([np.mean(s['collisions']) for s in sections])

    # --- Построение графиков ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) Гистограмма времени жизни для выбранных сечений
    ax = axes[0, 0]
    for idx in SELECTED_SECTIONS:
        if idx in by_index:
            s = by_index[idx]
            ax.hist(s['times']*1e3, bins=BINS, alpha=0.6,
                    edgecolor='black', label=f'сечение {idx} (x={s["x"]:.3g} м)')
    ax.set_xlabel('Время жизни, мс')
    ax.set_ylabel('Число частиц')
    ax.set_title('Гистограмма времени жизни')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Гистограмма числа столкновений для выбранных сечений
    ax = axes[0, 1]
    for idx in SELECTED_SECTIONS:
        if idx in by_index:
            s = by_index[idx]
            ax.hist(s['collisions'], bins=BINS, alpha=0.6,
                    edgecolor='black', label=f'сечение {idx} (x={s["x"]:.3g} м)')
    ax.set_xlabel('Число столкновений со стенкой, шт')
    ax.set_ylabel('Число частиц')
    ax.set_title('Гистограмма числа столкновений')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Среднее время жизни в зависимости от координаты сечения
    ax = axes[1, 0]
    ax.plot(l_all, mean_time_all*1e3, 'o-', color='tab:blue')
    ax.set_xlabel('Координата сечения l, м')
    ax.set_ylabel('Среднее время жизни, мс')
    ax.set_title('Среднее время жизни по сечениям')
    ax.grid(True, alpha=0.3)

    # 4) Среднее число столкновений в зависимости от координаты сечения
    ax = axes[1, 1]
    ax.plot(l_all, mean_coll_all, 's-', color='tab:red')
    ax.set_xlabel('Координата сечения l, м')
    ax.set_ylabel('Среднее число столкновений, шт')
    ax.set_title('Среднее число столкновений по сечениям')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Анализ данных по сечениям нейтрализатора')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


def plot_styled():
    exit()


    figures = {}
    axes = {}
    fontdict_labels = {'fontsize':26,
                'fontfamily': 'Times New Roman'}
    fontdict_ticks = {'fontsize':24,
                    'fontfamily': 'Times New Roman'}
    fontdict_title = {'fontsize':28,
                    'fontfamily': 'Times New Roman'}

    idx = np.ceil(section_amount/2)-1

    for image_idx in range(1, 5):
        figures[f'fig_{image_idx}'] = plt.figure(figsize=(13, 9))
        axes[f'ax_{image_idx}'] = figures[f'fig_{image_idx}'].add_subplot(111)

    # 1) Гистограмма времени жизни
    s = by_index[idx]
    axes[f'ax_{1}'].hist(s['times']*1e3, bins=BINS, alpha=0.6,
            edgecolor='black')
    axes[f'ax_{1}'].set_xlabel('Время жизни, мс', fontdict_labels)
    axes[f'ax_{1}'].set_ylabel('Число частиц', fontdict_labels)
    axes[f'ax_{1}'].set_title(f'Координата по нейтрализатору = {s['x']} м', fontdict_title)
    axes[f'ax_{1}'].grid(True, alpha=0.3)
    xticks = axes[f'ax_{1}'].get_xticks()
    axes[f'ax_{1}'].set_xticklabels([f'{t:.1f}' for t in xticks], fontdict = fontdict_ticks)
    yticks = axes[f'ax_{1}'].get_yticks()
    axes[f'ax_{1}'].set_yticklabels([f'{t:.1f}' for t in yticks], fontdict = fontdict_ticks)
    figures[f'fig_{1}'].show()
    input()
