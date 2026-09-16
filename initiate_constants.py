__all__ = ['M',
           'qe',
           'L',
           'a',
           'b',
           'a1',
           'b1',
           'alpha',
           'beta',
           'N',
           'filepath_short',
           'filepath_long',
           'filename_velocity',
           'filename_angle']

## Физические
import numpy as np

M = 1.67e-27
qe = 1.602e-19

## Геометрия
L = 2
a = 0.16
b = 0.47
a1 = 0.12
b1 = 0.35

## Параметры пучка
alpha = np.deg2rad(0.1) # Расходимость вдоль щелей
beta = np.deg2rad(0.5) # Расходимость поперек щелей

## Параметры расчета
N = 100 # Число модельных частиц, запускаемых из одного сечения
place = 'KI'
match place:
    case 'home':
        filepath_long = 'D:/PostgraduateStudy/MCC_neutral_15.09/props/profile_long.txt'
        filepath_short = 'D:/PostgraduateStudy/MCC_neutral_15.09/props/profile_short.txt'
        filename_velocity = 'D:/PostgraduateStudy/MCC_neutral_15.09/props/profile_energy.txt'
        filename_angle = 'D:/PostgraduateStudy/MCC_neutral_15.09/props/profile_angle.txt'
    case 'KI':
        filepath_long = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_long.txt'
        filepath_short = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_short.txt'
        filename_velocity = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_energy.txt'
        filename_angle = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_angle.txt'