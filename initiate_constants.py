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
           'filename_angle',
           'testprofile1',
           'testprofile2',
           'testprofile3',
           'section_amount']

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
N = 10 # Число модельных частиц, запускаемых из одного сечения
section_amount = 2 + 1 # Количество сечений по длине
place = 'KI'
match place:
    case 'home':
        filepath_long = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/profile_long.txt'
        filepath_short = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/profile_short.txt'
        filename_velocity = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/profile_energy.txt'
        filename_angle = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/profile_angle.txt'
        testprofile1 = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/dissociation_50kV_theta10.txt'
        testprofile2 = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/dissociation_50kV_theta50.txt'
        testprofile3 = 'D:/PostgraduateStudy/MCC_neutral_15.09/MCC_neutral_fraction/props/dissociation_50kV_theta90.txt'
    case 'KI':
        filepath_long = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_long.txt'
        filepath_short = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_short.txt'
        filename_velocity = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_energy.txt'
        filename_angle = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/profile_angle.txt'
        testprofile1 = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/dissociation_50kV_theta10.txt'
        testprofile2 = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/dissociation_50kV_theta50.txt'
        testprofile3 = 'D:/PostgraduateStudy/MCC_neutral_fraction/props/dissociation_50kV_theta90.txt'