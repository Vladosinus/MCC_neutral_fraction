from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

qe = 1.602e-19
me = 9.1e-31

## Ввод сечений и их легенда
# Для протонов  ==================================================================================================

sigma_proton_0 = 2.0568e-20;
sigma_proton_1 = 1.1796e-20 + 4.4223e-22 + 1.312e-21;

######

sigma_electron_ionization_molecular = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecules_dissociation_with_two_ions_by_proton.txt')
E = sigma_electron_ionization_molecular[:, 0]
sigma = sigma_electron_ionization_molecular[:, 1]
sigma_interpolator = interp1d(E, sigma, fill_value = 'extrapolate', bounds_error=False)
sigma_proton_2 = sigma_interpolator(60000)

sigma_proton_3 = 1.6683e-21 + 1.2019e-22 + 8.0413e-22;
sigma_proton_4 = 1.38e-20
sigma_proton_5 = 5.883e-21

sigma_electron_ionization_molecular = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecular_ion_dissociation_with_one_ions_by_proton.txt')
E = sigma_electron_ionization_molecular[:, 0]
sigma = sigma_electron_ionization_molecular[:, 1]
sigma_interpolator = interp1d(E, sigma, kind = 'cubic', fill_value = 'extrapolate', bounds_error=False)
sigma_proton_6 = sigma_interpolator(60000)
# Тут проблемы с экстраполяцией, поэтому все загружено, но сечение по графику найдено руками
sigma_proton_6 = 3.2e-20

# plt.plot(E, sigma, lw = 2, color = "#FF00E1", marker = 'o')
# plt.xlim([0, 65000])
# plt.show()


## Расшифровка сечений
# sigma_proton_0 Ионизация молекулы
# sigma_proton_1 CX на молекуле
# sigma_proton_2 Диоссоциация молекулы с образованием двух ионов
# sigma_proton_3 Диссоциация молекулы с образованием двух атомов
# sigma_proton_4 Ионизация атома
# sigma_proton_5 Перезарядка на атоме
# sigma_proton_6 Диссоциация молекулярного иона с образованием атома и иона
proton_description = (
    f"sigma_proton_0 Ионизация молекулы\n"
    f"sigma_proton_1 CX на молекуле\n"
    f"sigma_proton_2 Диоссоциация молекулы с образованием двух ионов\n"
    f"sigma_proton_3 Диссоциация молекулы с образованием двух атомов\n"
    f"sigma_proton_4 Ионизация атома\n"
    f"sigma_proton_5 Перезарядка на атоме\n"
    f"sigma_proton_6 Диссоциация молекулярного иона с образованием атома и иона\n"
                        )

with open('proton_sigmas.txt', 'w', encoding='utf-8') as f:
    f.write(proton_description)
    f.write(f'\n')
    
    f.write(f'{sigma_proton_0:.3e}\n')
    f.write(f'{sigma_proton_1:.3e}\n')
    f.write(f'{sigma_proton_2:.3e}\n')
    f.write(f'{sigma_proton_3:.3e}\n')
    f.write(f'{sigma_proton_4:.3e}\n')
    f.write(f'{sigma_proton_5:.3e}\n')
    f.write(f'{sigma_proton_6:.3e}\n')


# Для атомов  ==================================================================================================
sigma_neutr_0 = 1.3498e-20;
sigma_neutr_1 = 1.1811e-20;
sigma_neutr_2 = 9e-23 + 10e-23;
sigma_neutr_3 = 0
sigma_neutr_4 = 0.6e-21;
sigma_neutr_5 = 9e-23;
sigma_neutr_6 = 1.6e-21;
sigma_neutr_7 = 0;
sigma_neutr_8 = 0;

## Расшифровка сечений
# sigma_neutr_0 Ионизация молекулы
# sigma_neutr_1 Ионизация атома пучка
# sigma_neutr_2 Диссоциация молекулы с образованием одного иона и одного атома
# sigma_neutr_3 Диссоациация молекулы с образованием двух атомов
# sigma_neutr_4 Диссоциация молекулы с образованием двух ионов и ионизация атома пучка
# sigma_neutr_5 Диссоциация молекулы с образовнием одного атома и одного иона, ионизация атома пучка
# sigma_neutr_6 Диссоциация молекулы с образованием двух ионов
# sigma_neutr_7 Ионизация атома
# sigma_neutr_8 Диссоциация молекулярного иона с образованием атома и иона

neutral_description = (
    f"sigma_neutr_0 Ионизация молекулы\n"
    f"sigma_neutr_1 Ионизация атома пучка\n"
    f"sigma_neutr_2 Диссоциация молекулы с образованием одного иона и одного атома\n"
    f"sigma_neutr_3 Диссоациация молекулы с образованием двух атомов\n"
    f"sigma_neutr_4 Диссоциация молекулы с образованием двух ионов и ионизация атома пучка\n"
    f"sigma_neutr_5 Диссоциация молекулы с образовнием одного атома и одного иона, ионизация атома пучка\n"
    f"sigma_neutr_6 Диссоциация молекулы с образованием двух ионов\n"
    f"sigma_neutr_7 Ионизация атома\n"
    f"sigma_neutr_8 Диссоциация молекулярного иона с образованием атома и иона\n"
)

with open('neutral_sigmas.txt', 'w', encoding='utf-8') as f:
    f.write(neutral_description)
    f.write(f'\n')
    
    f.write(f'{sigma_neutr_0:.3e}\n')
    f.write(f'{sigma_neutr_1:.3e}\n')
    f.write(f'{sigma_neutr_2:.3e}\n')
    f.write(f'{sigma_neutr_3:.3e}\n')
    f.write(f'{sigma_neutr_4:.3e}\n')
    f.write(f'{sigma_neutr_5:.3e}\n')
    f.write(f'{sigma_neutr_6:.3e}\n')
    f.write(f'{sigma_neutr_7:.3e}\n')
    f.write(f'{sigma_neutr_8:.3e}\n')


## Для электронов ==================================================================================================
nrg = np.linspace(0, 200, 600)
velocity = np.sqrt(2*nrg*qe/me)

# Загружаем и интерполируем сечение на заданной сетке по энергии
sigma_electron_ionization_molecular = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecules_ionization_by_electron.txt')
E = sigma_electron_ionization_molecular[:, 0]
vnrg = np.sqrt(2*E*qe/me)
sigma_molecular = sigma_electron_ionization_molecular[:, 1]

sigma_molecular_interpolator = interp1d(E, sigma_molecular, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_molecular = sigma_molecular_interpolator(nrg)



sngr_molecular_interpolator = interp1d(vnrg, sigma_molecular, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_vel_molecular = sngr_molecular_interpolator(velocity)

del sngr_molecular_interpolator
del sigma_molecular_interpolator

sigma_electron_0 = sigma_electron_ionization_molecular
sigma_electron_0_vel = sigma_electron_ionization_vel_molecular




#####

sigma_electron_ionization_atomar = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Atoms_ionization_by_electron.txt')
E = sigma_electron_ionization_atomar[:, 0]
vnrg = np.sqrt(2*E*qe/me)
sigma_atomar = sigma_electron_ionization_atomar[:, 1]
sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

del sngr_atomar_interpolator
del sigma_atomar_interpolator

sigma_electron_1 = sigma_electron_ionization_atomar
sigma_electron_1_vel = sigma_electron_ionization_vel_atomar


#######

sigma_electron_ionization_atomar = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecular_ion_dissociation_with_one_ion_and_ionization_second_atom_by_electron.txt')
E = sigma_electron_ionization_atomar[:, 0]
vnrg = np.sqrt(2*E*qe/me)
sigma_atomar = sigma_electron_ionization_atomar[:, 1]
sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

del sngr_atomar_interpolator
del sigma_atomar_interpolator

sigma_electron_2 = sigma_electron_ionization_atomar
sigma_electron_2_vel = sigma_electron_ionization_vel_atomar


#######

sigma_electron_ionization_atomar = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecular_ion_dissociation_with_one_atom_and_second_ion_neutralization_by_electron.txt')
E = sigma_electron_ionization_atomar[:, 0]
vnrg = np.sqrt(2*E*qe/me)
sigma_atomar = sigma_electron_ionization_atomar[:, 1]
sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

del sngr_atomar_interpolator
del sigma_atomar_interpolator

sigma_electron_3 = sigma_electron_ionization_atomar
sigma_electron_3_vel = sigma_electron_ionization_vel_atomar

# Это сечение очень странное, оно максимально при 0,1 эВ, пусть оно пока нулем побудет

#######

sigma_electron_ionization_atomar = np.loadtxt('D:/PostgraduateStudy/Analytic_17_06_26/cross_section/Molecular_ion_dissociation_with_one_ion_and_one_atom_by_electron.txt')
E = sigma_electron_ionization_atomar[:, 0]
vnrg = np.sqrt(2*E*qe/me)
sigma_atomar = sigma_electron_ionization_atomar[:, 1]
sigma_atomar_interpolator = interp1d(E, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_atomar = sigma_atomar_interpolator(nrg)

sngr_atomar_interpolator = interp1d(vnrg, sigma_atomar, fill_value = 0, bounds_error=False, kind = 'cubic')
sigma_electron_ionization_vel_atomar = sngr_atomar_interpolator(velocity)

del sngr_atomar_interpolator
del sigma_atomar_interpolator

sigma_electron_4 = sigma_electron_ionization_atomar
sigma_electron_4_vel = sigma_electron_ionization_vel_atomar


## Расшифровка сечений
# sigma_electron_0 Ионизация молекулы
# sigma_electron_1 Ионизация атома
# sigma_electron_2 Диссоциация молекулярного иона с рождением иона и ионизацией атома
# sigma_electron_3 Диссоциация молекулярного иона с нейтрализацией иона и образованием атома
# sigma_electron_4 Диссоциация молекулярного иона с образованием атома и иона


# plt.plot(nrg, sigma_electron_3)
# plt.show()