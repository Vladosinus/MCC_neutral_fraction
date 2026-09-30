import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from initiate_constants import *

filepath = 'output/mean_vals.txt'

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

sections = read_section_data(filepath)
l = np.linspace(0, L, len(sections))
drift_speed = np.zeros(len(sections)-1)

for idx in range(len(sections)-1):
    drift_speed[idx] = (L - sections[idx]['x'])/sections[idx]['mean_time']
l1 = l[:-1]





times = np.zeros(len(sections))
for i in range(len(sections)):
    times[i] = sections[i]['mean_time']
vx = (L - l[:-2])/times[:-2]

extrapolator = interp1d(l[:-2], vx, kind = 'linear', fill_value='extrapolate', bounds_error=False)
v_drift = extrapolator(l)

D = 1/3*v_drift*a*k













# plt.plot(l, vx1)
# plt.ylim([0, 5000])
# plt.show()

exit()






















l_interp = np.linspace(0, L, 200)

interpolator = interp1d(l, times, kind='cubic', bounds_error=False)
times_interp = interpolator(l_interp)
v = (L - l_interp)/times_interp

plt.plot(l_interp, v)
plt.ylim([0, 5000])
plt.show()




# plt.plot(l_interp, times_interp)
# plt.plot(l, times)
# plt.show()


exit()
plt.plot(l1[:-1], sections[:-2]['mean_time'])
plt.show()