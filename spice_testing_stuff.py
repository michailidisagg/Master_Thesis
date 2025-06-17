import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import sunpy.map
import sunpy_soar
from astropy.io import fits
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from sunraster.instr.spice import read_spice_l2_fits
import os
from moviepy.editor import ImageSequenceClip
from sunpy.coordinates import frames
from sunraster.instr.spice import read_spice_l2_fits
from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval, ImageNormalize
from scipy.stats import norm
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
#-----------------------------------------------------------------------------------------------------------------------
#Get input spice fits data
spice_folder = 'C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\spice\\2022_03_03'
spice_files = sorted([os.path.join(spice_folder, f) for f in os.listdir(spice_folder) if f.endswith('.fits')])
#spice_files = ["deconvolved_spice_ne_viii_cube.fits"]
#-----------------------------------------------------------------------------------------------------------------------
#Open and edit/compute each fits file
for frame in range(0,1):
    print(f'Computing image {frame+1}/{len(spice_files)}') #monitoring
    hdulist = fits.open(spice_files[frame])
    data = hdulist[0].data
    header = hdulist[0].header
    hdulist.close
    #extract info from header
    spectral_line =  header['EXTNAME']
    print(spectral_line)
    wavemin = header['WAVEMIN']
    wavemax = header['WAVEMAX']
    winwidth = header['WINWIDTH']
    spectrum_range = np.arange(wavemin, wavemax, (wavemax-wavemin)/data.shape[1])
    date_obs = header['DATE-OBS']
    int_unit = header['BUNIT']
    # Create/set output folder for movie
    #output_folder_image = f"C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\spice\\2022_03_03\\image_{spectral_line}_pngs"
    #os.makedirs(output_folder_image, exist_ok=True)
    #output_folder_spectrum = f"C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\spice\\2022_03_03\\spectrum_{spectral_line}_pngs"
    #os.makedirs(output_folder_spectrum, exist_ok=True)
    # -----------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #Integrade the spectrum to make integrated image
    image = np.zeros((data.shape[2],data.shape[3]))
    for i in range(data.shape[1]):
        image = image + np.nan_to_num(np.array(data[0,i,:,:]),0) #make data np and then replace nan with 0
    #-------------------------------------------------------------------------------------------------------------------
    #Integrade image to get spectrum
    spectrum = np.zeros(data.shape[1])
    image_fit = np.zeros((data.shape[2],data.shape[3]))
    doppler_fit = np.zeros((data.shape[2], data.shape[3]))
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):

            spectrum = np.nan_to_num(np.array(data[0, :, i, j]), 0)
    #        # Apply Gaussian fit
            trim = min(len(spectrum_range), len(spectrum))  # make sure they have the smae size
            spectrum_range = spectrum_range[:trim]
            spectrum = spectrum[:trim]
            n = len(spectrum)

            # My initial guesses for the fitting
            # I Assume 4 spectral line (3 O III lines and the Mg Ix line)
            #Actually for my thesis I only fitted the main gaussian because the 4-gaussian fit takes 1.5hours to run, and doesn't affect what happens at the center
            #of the spectral line, but the code for the multiple gaussian fit is here
            mean1 = spectrum_range[np.argmin(np.abs(spectrum_range - 77.04))]
            sigma1 = spectrum_range[2] - spectrum_range[0]
            amplitude1 = spectrum[np.argmin(np.abs(spectrum_range - 77.04))]

            #mean2 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.28186359))]
            #sigma2 = spectrum_range[2] - spectrum_range[0]
            #amplitude2 = spectrum[np.argmin(np.abs(spectrum_range - 70.28186359))]

            #mean3 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.38913229))]
            #sigma3 = spectrum_range[2] - spectrum_range[0]
            #amplitude3 = spectrum[np.argmin(np.abs(spectrum_range - 70.38913229))]

            #mean4 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.60366969))]
            #sigma4 = spectrum_range[2] - spectrum_range[0]
            #amplitude4 = spectrum[np.argmin(np.abs(spectrum_range - 70.60366969))]
            baseline = np.min(spectrum)  # Assume lowest value is background noise
            #def gaus(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3, a4, x04, sigma4, b):
                #return b + a1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + a2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2)) + a3 * np.exp(-(x - x03) ** 2 / (2 * sigma3 ** 2)) + a4 * np.exp(-(x - x04) ** 2 / (2 * sigma4 ** 2))

            def gaus(x, a1, x01, sigma1, b):
                return b + a1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2))


            # Always clip negative values
            spectrum = np.clip(spectrum, 0, None)

            #try:
            #    # Fit the Gaussian to the spectrum
            #    popt, pcov = curve_fit(gaus, spectrum_range, spectrum, p0=[amplitude1, mean1, sigma1, baseline])
            #    spectrum_fit = gaus(spectrum_range, *popt)
            #    image_fit[i, j] = popt[0] * popt[2] * 2 * np.pi
            #    doppler_fit[i, j] = popt[1]
            #except RuntimeError:
            #    print(f"Fit failed at pixel ({i},{j}), using default values.")
            #    image_fit[i, j] = 0
            #    doppler_fit[i, j] = 77.04

    #sunpy.map.Map(image_fit, header).save('image_fit_ne_viii_deconvolved.fits', overwrite=True)
    #sunpy.map.Map(doppler_fit, header).save('doppler_fit_ne_viii_deconvolved.fits', overwrite=True)
    #print('i saved the fits bruh')
    # Integrade image to get spectrum
    spectrum = np.zeros(data.shape[1])
    for i in range(550,650):
        for j in range(145,165):
            print(i,j, np.nan_to_num(np.array(data[0, :, i, j]), 0))
            spectrum = spectrum + np.nan_to_num(np.array(data[0, :, i, j]), 0)
    #-------------------------------------------------------------------------------------------------------------------
    #Apply Gaussian fit
    trim = min(spectrum_range.shape[0], spectrum.shape[0]) #make sure they have the smae size
    spectrum_range = spectrum_range[:trim]
    spectrum = spectrum[:trim]
    n = len(spectrum)
    mean1 = spectrum_range[np.argmin(np.abs(spectrum_range -  77.04))]
    sigma1 = spectrum_range[2] - spectrum_range[0]
    amplitude1 = spectrum[np.argmin(np.abs(spectrum_range - 77.04))]

    mean2 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.28186359))]
    sigma2 = spectrum_range[2] - spectrum_range[0]
    amplitude2 = spectrum[np.argmin(np.abs(spectrum_range - 70.28186359))]

    mean3 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.38913229))]
    sigma3 = spectrum_range[2] - spectrum_range[0]
    amplitude3 = spectrum[np.argmin(np.abs(spectrum_range - 70.38913229))]

    mean4 = spectrum_range[np.argmin(np.abs(spectrum_range - 70.60366969))]
    sigma4 = spectrum_range[2] - spectrum_range[0]
    amplitude4 = spectrum[np.argmin(np.abs(spectrum_range - 70.60366969))]

    baseline = np.min(spectrum)  # Assume lowest value is background noise


    def gaus(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3, a4, x04, sigma4, b):
        return b + a1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + a2 * np.exp(
            -(x - x02) ** 2 / (2 * sigma2 ** 2)) + a3 * np.exp(-(x - x03) ** 2 / (2 * sigma3 ** 2)) + a4 * np.exp(
            -(x - x04) ** 2 / (2 * sigma4 ** 2))
    def gaus1(x, a1, x01, sigma1,b):
        return b + a1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2))

    popt, pcov = curve_fit(gaus1, spectrum_range, spectrum, p0=[amplitude1, mean1, sigma1, baseline])
    #popt, pcov = curve_fit(gaus, spectrum_range, spectrum, p0=[amplitude1, mean1, sigma1,amplitude2, mean2, sigma2,amplitude3, mean3, sigma3,amplitude4, mean4, sigma4, baseline])
    #Im = popt[0]
    mean = popt[1]
    print(mean)
    #sigma = popt[2]
    #err_Im = np.sqrt(pcov[0,0])
    #err_sigma = np.sqrt(pcov[2,2])
    #err_It = 2*np.pi*np.sqrt((Im*err_sigma)**2+(sigma*err_Im)**2)
    #print(2*np.pi*Im*sigma,err_It)
    # Plot image
   # fig = plt.figure(figsize=(16,9))
   # plt.imshow(image, aspect='auto', origin='lower', cmap='gray', vmin=np.percentile(image, 1),
   #            vmax=np.percentile(image, 99))
   # plt.title(f'{spectral_line}, {date_obs}')
   # plt.xlabel('x pixels')
   # plt.ylabel('y pixels')
   # plt.grid(alpha=0.5, linestyle='--')
   # plt.tight_layout()
   # #frame_filename = os.path.join(output_folder_image, f'frame_{frame:03d}.png')
   # #plt.savefig(frame_filename)
    fig = plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.imshow(np.log(image_fit), aspect='auto', origin='lower', cmap='gray',vmin=np.percentile(image_fit, 1),
               vmax=np.percentile(image_fit, 99))
    plt.title(f'{spectral_line}, {date_obs} Intensity')
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.subplot(122)
    plt.imshow(doppler_fit-np.mean(doppler_fit), aspect='auto', origin='lower', cmap='bwr')
    plt.title(f'{spectral_line}, {date_obs} Doppler Map')
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    # Plot spectrum
    plt.figure(figsize=(16,9))
    plt.plot(spectrum_range, spectrum, label=f'Data {date_obs}')
    plt.plot(np.linspace(np.min(spectrum_range), np.max(spectrum_range),1000),gaus1(np.linspace(np.min(spectrum_range), np.max(spectrum_range),1000),*popt), color='red', linestyle='--', label='Fit')
    plt.title(f'{spectral_line}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(f'Intensity ({int_unit})')
    plt.grid()
    plt.tight_layout()
    plt.legend()

#-----------------------------------------------------------------------------------------------------------------------
#Display the plots
plt.show()
#create movie
#image_files_image = [os.path.join(output_folder_image, f) for f in sorted(os.listdir(output_folder_image)) if f.endswith('.png')]
#clip = ImageSequenceClip(image_files_image, fps=1)
#clip.write_videofile(f"{output_folder_image}\\spice_image.mp4", codec="libx264")