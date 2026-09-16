import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


from initiate_constants import *

def find_beam_profile(l):
    a2 = 2*l*np.tan(alpha) + a1
    b2 = 2*l*np.tan(beta) + b1
    return a2, b2

def distibute_particles(a1, b1, a_middle, b_middle, N):
    stretching_koef_a = a_middle/a1
    
    # Короткая сторона
    x_a = np.linspace(-a1/2, a1/2, 100)
    temp_a = np.loadtxt(filepath_short)
    temp_a[0] = -a1/2
    temp_a[-1] = a1/2
    density_a = interpolate.interp1d(temp_a[:, 0], temp_a[:, 1], kind = 'linear', bounds_error = False, fill_value = 0)
    y_a = density_a(x_a)
    y_a = y_a / np.trapezoid(y_a, x_a)
    x_a = x_a*stretching_koef_a

    cdf = np.zeros_like(x_a)
    for i in range(1, len(x_a)):
        cdf[i] = cdf[i-1] + np.trapezoid(y_a[i-1:i+1], x_a[i-1:i+1])
    cdf = cdf / cdf[-1]
    
    cdf_inverse = interpolate.interp1d(cdf, x_a, kind='linear', bounds_error=False, fill_value='extrapolate')
    rng = np.random.default_rng(seed=42)
    u_a = rng.random(N)
    s_a = cdf_inverse(u_a)


    # Длинная сторона
    stretching_koef_b = b_middle/b1
    x_b = np.linspace(-b1/2, b1/2, 100)
    temp_b = np.loadtxt(filepath_long)
    temp_b[0] = -b1/2
    temp_b[-1] = b1/2
    density_b = interpolate.interp1d(temp_b[:, 0], temp_b[:, 1], kind = 'linear', bounds_error = False, fill_value = 0)
    y_b = density_b(x_b)
    y_b = y_b / np.trapezoid(y_b, x_b)
    x_b = x_b*stretching_koef_b

    cdf = np.zeros_like(x_b)
    for i in range(1, len(x_b)):
        cdf[i] = cdf[i-1] + np.trapezoid(y_b[i-1:i+1], x_b[i-1:i+1])
    cdf = cdf / cdf[-1]
    from scipy.interpolate import interp1d
    cdf_inverse = interp1d(cdf, x_b, kind='linear', bounds_error=False, fill_value='extrapolate')
    rng = np.random.default_rng(seed=42)
    u_b = rng.random(N)
    s_b = cdf_inverse(u_b)

    return s_a, s_b

def calculate_velocity(mean_energy, mean_angle):
            
    speed = (2*mean_energy*qe/M)**0.5
    
    ## Назначаем скорости
    vx0 = np.zeros(N)
    
    vx0[:] = speed*np.abs(np.cos(np.deg2rad(mean_angle)))
    v_transverse = speed*np.abs(np.sin(np.deg2rad(mean_angle)))
    return vx0, v_transverse

def distribute_velocity_projections(N, v_transverse):

    vy0 = np.zeros(N)
    vz0 = np.zeros(N)
    
    for n in range(N):
        gamma = np.random.rand()*360
        if gamma >= 0 and gamma < 90:
            vy0[n] = v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = -v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 90 and gamma < 180:
            vy0[n] = -v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = -v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 180 and gamma < 270:
            vy0[n] = -v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = v_transverse*np.abs(np.sin(np.deg2rad(gamma)))
        elif gamma >= 270 and gamma < 360:
            vy0[n] = v_transverse*np.abs(np.cos(np.deg2rad(gamma)))
            vz0[n] = v_transverse*np.abs(np.sin(np.deg2rad(gamma)))

    
    return vy0, vz0

def collision_counter(time, solutiuon):
    # Распаковываем решение
    x = solutiuon[0, :]        
    y = solutiuon[2, :]
    vy = solutiuon[3, :]
    z = solutiuon[4, :]
    vz = solutiuon[5, :]

    # По изменению знака скорости будем считать, сколько столкновений с какой стенкой частица совершила
    # Соударения с вертикальными стенками
    sign_changes = np.where(np.diff(np.sign(vy)))[0]
    num_crossings_vertical = len(sign_changes)
    sign_changes = np.where(np.diff(np.sign(vz)))[0]
    num_crossings_horizontal = len(sign_changes)
    total_collisions = num_crossings_vertical + num_crossings_horizontal
    plt.plot(x, y)
    plt.show()
    return total_collisions

def section_treater():
    # Здесь будет найдена средняя энергия частиц и средний угол вылета, чтобы потом их отправить в calculate_velocity
    
    secondary_energy = np.linspace(0, 20, 100)
    ## 10 градусов
    theta10 = np.loadtxt(testprofile1)
    nrg1 = theta10[:, 0]
    sig = theta10[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=15, polyorder=3)
    # Интерполяция на общую сетку
    theta10_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma10 = theta10_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma10[sigma10 < 0] = 0

    ## 50 градусов
    theta50 = np.loadtxt(testprofile2)
    nrg1 = theta50[:, 0]
    sig = theta50[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=11, polyorder=3)
    # Интерполяция на общую сетку
    theta50_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma50 = theta50_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma50[sigma50 < 0] = 0

    ## 90 градусов
    theta90 = np.loadtxt(testprofile3)
    nrg1 = theta90[:, 0]
    sig = theta90[:, 1]
    # Поиск дубликатов
    unique_nrg1, indices = np.unique(nrg1, return_index=True)
    unique_sig = sig[indices]
    # Сглаживание
    sig_smooth = savgol_filter(unique_sig, window_length=11, polyorder=3)
    # Интерполяция на общую сетку
    theta90_interpolator = interpolate.interp1d(unique_nrg1, sig_smooth, kind = 'cubic', fill_value = 0, bounds_error = False)
    sigma90 = theta90_interpolator(secondary_energy)
    # Артефакты после сглаживания
    sigma90[sigma90 < 0] = 0,

    ## 0 градусов
    # Искусственно создадим такое распрееление, так как измнрить его невозможно
    sigma0 = sigma10*1.05

    thetas = [0, 10, 50, 90]

    sigmas = np.column_stack((sigma0, sigma10, sigma50, sigma90))
    
    ## Средняя энергия
    only_energy_distibution = np.zeros(len(sigmas[:, 0]))
    for i in range(len(sigmas[:, 0])):
        only_energy_distibution[i] = np.trapezoid(sigmas[i, :], thetas)

    # print(len(sigmas[:, 0]))
    # print(len(secondary_energy))
    # exit()

    ## Средний угол
    only_angle_distibution = np.zeros(len(sigmas[0, :]))
    for i in range(len(sigmas[0, :])):
        only_angle_distibution[i] = np.trapezoid(sigmas[:, i], secondary_energy)

    #  Находим средние значения энергии и угла
    mean_energy = np.trapezoid(only_energy_distibution*secondary_energy, secondary_energy)/np.trapezoid(only_energy_distibution, secondary_energy)
    mean_angle = np.trapezoid(only_angle_distibution*thetas, thetas)/np.trapezoid(only_angle_distibution, thetas)
       
    return mean_energy, mean_angle

def diffuse_deflection():
    # Азимутальный угол
    azimuth = np.random.rand()*360

    # Нормальный угол
    normal = np.random.rand()*180
    
    return azimuth, normal

