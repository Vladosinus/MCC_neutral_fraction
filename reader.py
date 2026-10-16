"""Чтение текстовых данных, сохранённых в MCC_recombination.py."""

import ast
from pathlib import Path
import re

import numpy as np


DEFAULT_FILE = Path(__file__).resolve().with_name('MCC_traj.txt')
FIELDS = ('time', 'trajectories', 'collisions_count', 'collision_points')


def _decode_node(node):
    """Разбирает только числа, контейнеры и известные обёртки NumPy."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_decode_node(item) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _decode_node(node.operand)
        if type(value) not in (int, float):
            raise ValueError('Знак допустим только перед числом.')
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id in ('array', 'list'):
            value = _decode_node(node.args[0])
            if not isinstance(value, (list, tuple)):
                raise ValueError('Ожидался список значений.')
            return np.array(value, dtype=float) if node.func.id == 'array' else list(value)
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'np'
                and node.func.attr == 'float64'):
            return float(_decode_node(node.args[0]))
    raise ValueError('Недопустимое выражение в данных.')

def _parse_field(text, name):
    """Восстанавливает пропущенные при печати разделители без выполнения кода."""
    if '...' in text:
        raise ValueError('Массив сокращён многоточием: полные данные не сохранены.')
    if name == 'collisions_count':
        # Числовой массив печатается как [30. 26.], без запятых.
        if not (text.startswith('[') and text.endswith(']')):
            raise ValueError('Ожидался массив чисел столкновений.')
        values = np.array([float(v) for v in text[1:-1].replace(',', ' ').split()])
        if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
            raise ValueError('Число столкновений должно быть целым и неотрицательным.')
        return values
    # Внешний объектный массив печатается без запятых между array(...) / list(...).
    text = re.sub(r'\)\s+(?=(?:array|list)\()', '), ', text)
    try:
        result = _decode_node(ast.parse(text, mode='eval').body)
    except (SyntaxError, TypeError) as exc:
        raise ValueError('Неверный формат массива.') from exc
    if not isinstance(result, list):
        raise ValueError('Ожидался список данных по частицам.')
    return result


def _validate_cell(cell):
    """Проверяет полноту данных и соответствие размеров для каждой частицы."""
    if set(cell) != set(FIELDS):
        raise ValueError('Ячейка должна содержать все четыре блока данных.')
    count = len(cell['time'])
    if any(len(cell[name]) != count for name in FIELDS):
        raise ValueError('Количество частиц в блоках не совпадает.')
    for index in range(count):
        time = cell['time'][index]
        trajectory = cell['trajectories'][index]
        points = cell['collision_points'][index]
        if not isinstance(time, np.ndarray) or time.ndim != 1:
            raise ValueError(f'Частица {index}: время должно быть одномерным массивом.')
        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (len(time), 6):
            raise ValueError(f'Частица {index}: ожидалась траектория размером (число времён, 6).')
        if not isinstance(points, list) or len(points) != cell['collisions_count'][index]:
            raise ValueError(f'Частица {index}: число точек столкновений не совпадает со счётчиком.')
        for point in points:
            if (not isinstance(point, tuple) or len(point) != 3
                    or not isinstance(point[0], (int, float))
                    or not isinstance(point[1], np.ndarray) or point[1].shape != (6,)
                    or type(point[2]) is not int or point[2] not in range(1, 6)):
                raise ValueError(f'Частица {index}: неверная запись точки столкновения.')

def read_mcc_traj(filepath=DEFAULT_FILE):
    """Читает файл и возвращает словарь sections[section][(i, j)].

    В каждой ячейке находятся:
        time — список одномерных массивов времени по частицам;
        trajectories — список массивов (число точек, 6), столбцы x, vx, y, vy, z, vz;
        collisions_count — массив чисел столкновений;
        collision_points — списки кортежей (время, состояние из 6 чисел, индекс стенки).

    Индексы стенок: 1 — левая, 2 — правая, 3 — нижняя, 4 — верхняя,
    5 — входная. Списки сохраняют траектории разной длины без дополнения нулями.
    По умолчанию файл ищется рядом с reader.py, независимо от рабочей папки.
    Неполные, сокращённые или некорректные данные вызывают ValueError.
    """
    sections = {}
    section = None
    cell = None
    cell_indexes = None
    field = None
    lines = []

    def finish_field():
        if field is not None:
            try:
                cell[field] = _parse_field(''.join(lines).strip(), field)
            except ValueError as exc:
                raise ValueError(
                    f'Сечение {section}, ячейка {cell_indexes}, блок {field}: {exc}'
                ) from exc

    with open(filepath, encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, start=1):
            label = line.strip()
            if not label:
                continue
            section_match = re.fullmatch(r'section\s+(\d+)', label)
            cell_match = re.fullmatch(r'cell indexes\s+(\d+),\s*(\d+)', label)
            if section_match or cell_match or label in FIELDS:
                finish_field()
                field, lines = None, []
                if section_match:
                    section = int(section_match[1])
                    if section in sections:
                        raise ValueError(f'Сечение {section} указано повторно.')
                    sections[section] = {}
                    cell = None
                elif cell_match:
                    if section is None:
                        raise ValueError(f'Строка {line_number}: ячейка вне сечения.')
                    cell_indexes = (int(cell_match[1]), int(cell_match[2]))
                    if cell_indexes in sections[section]:
                        raise ValueError(f'Сечение {section}: повтор ячейки {cell_indexes}.')
                    cell = {}
                    sections[section][cell_indexes] = cell
                else:
                    if cell is None or label in cell:
                        raise ValueError(f'Строка {line_number}: неожиданный блок {label}.')
                    field = label
            elif field is not None:
                lines.append(line)
            else:
                raise ValueError(f'Строка {line_number}: неизвестный заголовок {label!r}.')
        finish_field()

    if not sections:
        raise ValueError('Файл не содержит сечений.')
    for section, cells in sections.items():
        if not cells:
            raise ValueError(f'Сечение {section} не содержит ячеек.')
        for indexes, cell in cells.items():
            try:
                _validate_cell(cell)
            except ValueError as exc:
                raise ValueError(f'Сечение {section}, ячейка {indexes}: {exc}') from exc
    return sections


if __name__ == '__main__':
    sections = read_mcc_traj()
    print(f'Прочитан файл: {DEFAULT_FILE}')
    print(f'Сечений: {len(sections)}')
    for section, cells in sections.items():
        particle_count = sum(len(cell['time']) for cell in cells.values())
        print(f'Сечение {section}: ячеек — {len(cells)}, частиц — {particle_count}')


    cell = sections[0][(0, 0)]          # секция 0, ячейка (0, 0)
    t = cell['time'][0]                 # время частицы 0
    traj = cell['trajectories'][0]      # форма (N, 6), столбцы x, vx, y, vy, z, vz
    n_coll = cell['collisions_count'][0]
    points = cell['collision_points'][0]  # кортежи (t, state, wall_index)

    print(t)


