__all__ = ['M',
           'M2',
           'kB',
           'qe',
           'T',
           'Q',
           'L',
           'a',
           'b',
           'a1',
           'b1',
           'k',
           'Dh',
           'alpha',
           'beta',
           'I_beam',
           'E_beam',
           'v_beam',
           'n_beam',
           'N',
           'filepath_short',
           'filepath_long',
           'filename_velocity',
           'filename_angle',
           'testprofile1',
           'testprofile2',
           'testprofile3',
           'section_amount',
           'use_multiprocessing',
           'n_workers']

## Физические
import numpy as np

# Физические константы
M = 1.67e-27
M2 = 2*M
kB = 1.38e-23
qe = 1.602e-19

# Просто параметры чего-нтбудь
T = 300 # Температура газа при напуске
Q = 3.3 # Полный напуск газа в нейтрализатор м^3*Па/с

## Геометрия
L = 2
a = 0.16
b = 0.47
a1 = 0.12
b1 = 0.35
k = 1.2 # Коэффициент для молекулярного течения
Dh = 4*a*b/(2*(a + b)) # Гидравлический диаметр

## Параметры пучка
alpha = np.deg2rad(0.1) # Расходимость вдоль щелей
beta = np.deg2rad(2) # Расходимость поперек щелей
I_beam = 40
E_beam = 60e3
v_beam = (2*E_beam*qe/M)**0.5
n_beam = I_beam/(qe*v_beam*a1*b1)


## Параметры расчета
N = 10000 # Число модельных частиц, запускаемых из одного сечения
section_amount = 20 # Количество сечений по длине

## Параметры распараллеливания
use_multiprocessing = True # Включить распараллеливание трассировки частиц
n_workers = 0 # Число рабочих процессов (0 — использовать os.cpu_count())

place = 'home'
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