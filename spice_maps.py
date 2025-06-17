import sunpy.map
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#For my thesis I use these files
#image_files = ['image_fit_ly_b.fits', 'image_fit_o_vi.fits','image_fit_ne_viii.fits', 'image_fit_c_iii.fits']
#doppler_files = ['doppler_fit_lyman-beta.fits', 'doppler_fit_o_vi.fits','doppler_fit_ne_viii_2.fits', 'doppler_fit_c_iii.fits']
#spectral_lines = ['Lyman-β', 'O VI','Ne VIII', 'C III']


#This is something I might try in the future
image_files = ['image_fit_ne_viii_deconvolved.fits']
doppler_files = ['doppler_fit_ne_viii_deconvolved.fits']
spectral_lines = ['Ne VIII']

# 2D polynomial surface (2nd-order) for better trend removal
def linear(x, a, b):
    return a * x + b

for image, doppler, line in zip(image_files, doppler_files, spectral_lines):
    # Load maps
    image_map = sunpy.map.Map(image)
    doppler_map = sunpy.map.Map(doppler)
    data = doppler_map.data.copy()

    region = data[0:700, :]
    #region = data

    # ---------------- X-axis Correction ----------------
    column_means_before = np.mean(region, axis=0)

    x = np.arange(column_means_before.size)
    popt_x, _ = curve_fit(linear, x, column_means_before)
    trend_x = linear(x, *popt_x)
    pattern_x = np.tile(trend_x, (data.shape[0], 1))
    corrected_data = data - pattern_x
    data = corrected_data

    # Optional re-zero-center
    # data = corrected_data - np.mean(corrected_data[200:650, :])
    # data = corrected_data - np.mean(corrected_data)
    region = data[0:700, :]
    # region = data
    # column_means_after = np.mean(region, axis=0)

    # ---------------- Y-axis Correction ----------------
    row_means_before = np.mean(region, axis=1)

    y = np.arange(row_means_before.size)
    popt_y, _ = curve_fit(linear, y, row_means_before)
    trend_y = linear(y, *popt_y)
    pattern_y = np.tile(trend_y[:, np.newaxis], (1, data.shape[1]))
    #data[200:650, :] -= pattern_y
    # data -= pattern_y

    #data = data - np.mean(data[200:650, :])
    print(np.median(data[200:700,:]), np.median(data[200:700,:]))
    #data = data - np.mean(data[200:650])
    #data = data - np.mean(data)
    #data = data - 77.042
    # region = data[200:650, :]
    # region = data
    # row_means_after = np.mean(region, axis=1)
    # ----- Plotting -----
    doppler_map = sunpy.map.Map(data, doppler_map.meta)
    image_map = sunpy.map.Map(np.log(image_map.data), image_map.meta)

    fig = plt.figure(figsize=(10, 5))

    ax_1 = fig.add_subplot(121, projection=image_map)
    if line == 'Ne VIII':
        image_map.plot(axes=ax_1, aspect='auto', origin='lower', vmin=-4, vmax=3)
    elif line == 'Lyman-β':
        image_map.plot(axes=ax_1, aspect='auto', origin='lower', vmin=-1, vmax=4)
    else:
        image_map.plot(axes=ax_1, aspect='auto', origin='lower', vmin=-3, vmax=3)
    ax_1.set_title(f'{line} Intensity')

    ax_2 = fig.add_subplot(122, projection=doppler_map)
    doppler_map.plot(axes=ax_2, aspect='auto',
                     vmin=-0.02, vmax=0.02,
                     cmap='bwr',
                     origin='lower')
    ax_2.set_title(f'{line} Doppler Shift')

    plt.tight_layout()
    plt.show()