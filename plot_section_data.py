"""Чтение данных моделирования из txt-файла и построение графиков.

Скрипт:
  1. Читает txt-файл, созданный auxillary.save_section_data.
  2. Строит гистограммы времени жизни и числа столкновений для выбранных
     сечений (два окна на одном графике).
  3. На том же figure строит два графика зависимостей от координаты l:
     среднее время жизни в сечении и среднее число столкновений со стенкой.
"""

import re
import numpy as np
import matplotlib.pyplot as plt


from initiate_constants import *


# Путь к файлу с данными
FILEPATH = 'section_data.txt'

# Номера сечений, для которых строятся гистограммы (индексы из файла)
SELECTED_SECTIONS = [5]

# Число бинов для гистограмм
BINS = 50


def read_section_data(filepath):
    """Читает txt-файл с данными по сечениям.

    Возвращает:
        sections - список словарей вида
                   {'index': k, 'x': x, 'times': np.array, 'collisions': np.array}
    """
    sections = []
    current = None

    section_re = re.compile(r'#\s*SECTION\s+(\d+)\s*\|\s*x\s*=\s*([-\d.eE+]+)')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = section_re.match(line.strip())
            if m:
                # Начался новый блок сечения
                if current is not None:
                    sections.append(current)
                current = {
                    'index': int(m.group(1)),
                    'x': float(m.group(2)),
                    'times': [],
                    'collisions': [],
                }
                continue

            # Строки данных: "  <n>  <time>  <collisions>"
            if current is not None:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                if len(parts) == 3:
                    try:
                        t = float(parts[1])
                        c = int(parts[2])
                    except ValueError:
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
    sections = read_section_data(FILEPATH)
    if not sections:
        print(f'Не удалось прочитать данные из {FILEPATH}')
        return

    # Индексация по номеру сечения
    by_index = {s['index']: s for s in sections}

    # # Массивы средних значений по всем сечениям (для графиков от l)
    # l_all = np.array([s['x'] for s in sections])
    # mean_time_all = np.array([np.mean(s['times']) for s in sections])
    # mean_coll_all = np.array([np.mean(s['collisions']) for s in sections])

    # # --- Построение графиков ---
    # fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # # 1) Гистограмма времени жизни для выбранных сечений
    # ax = axes[0, 0]
    # for idx in SELECTED_SECTIONS:
    #     if idx in by_index:
    #         s = by_index[idx]
    #         ax.hist(s['times']*1e3, bins=BINS, alpha=0.6,
    #                 edgecolor='black', label=f'сечение {idx} (x={s["x"]:.3g} м)')
    # ax.set_xlabel('Время жизни, мс')
    # ax.set_ylabel('Число частиц')
    # ax.set_title('Гистограмма времени жизни')
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # # 2) Гистограмма числа столкновений для выбранных сечений
    # ax = axes[0, 1]
    # for idx in SELECTED_SECTIONS:
    #     if idx in by_index:
    #         s = by_index[idx]
    #         ax.hist(s['collisions'], bins=BINS, alpha=0.6,
    #                 edgecolor='black', label=f'сечение {idx} (x={s["x"]:.3g} м)')
    # ax.set_xlabel('Число столкновений со стенкой, шт')
    # ax.set_ylabel('Число частиц')
    # ax.set_title('Гистограмма числа столкновений')
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # # 3) Среднее время жизни в зависимости от координаты сечения
    # ax = axes[1, 0]
    # ax.plot(l_all, mean_time_all*1e3, 'o-', color='tab:blue')
    # ax.set_xlabel('Координата сечения l, м')
    # ax.set_ylabel('Среднее время жизни, мс')
    # ax.set_title('Среднее время жизни по сечениям')
    # ax.grid(True, alpha=0.3)

    # # 4) Среднее число столкновений в зависимости от координаты сечения
    # ax = axes[1, 1]
    # ax.plot(l_all, mean_coll_all, 's-', color='tab:red')
    # ax.set_xlabel('Координата сечения l, м')
    # ax.set_ylabel('Среднее число столкновений, шт')
    # ax.set_title('Среднее число столкновений по сечениям')
    # ax.grid(True, alpha=0.3)

    # fig.suptitle('Анализ данных по сечениям нейтрализатора')
    # plt.tight_layout()
    # plt.show()
    # exit()

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
    from matplotlib.ticker import FuncFormatter
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

if __name__ == '__main__':
    main()
