import os
import ChiantiPy
os.environ['HOME'] = os.path.expanduser("~")
os.environ['XUVTOP'] = "C:\\Users\\ageli\\Desktop\\CHIANTI_11.0_database"
import ChiantiPy.core as ch
import ChiantiPy.tools.filters as chfilters
import ChiantiPy.tools.io as chio
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------
# Set atmospheric parameters
temp = 10.**(5.0 + (7.5-5)/30*np.arange(29.)) #match DEM from AIA
dens = 1.e+9 #reference value
#-----------------------------------------------------------------------------------------------------------------------
# Contribution function of Ne VIII
ne_8 = ch.ion('ne_8', temp, dens)
ne_8.intensity()
dist = np.abs(np.asarray(ne_8.Intensity['wvl']) - 770.4280)
idx = np.argmin(dist)
cfne8 = ne_8.Intensity['intensity'][:,idx]
#-----------------------------------------------------------------------------------------------------------------------
# Contribution function of Mg IX
mg_9 = ch.ion('mg_9', temp, dens)
mg_9.intensity()
dist = np.abs(np.asarray(mg_9.Intensity['wvl']) - 706.)
idx = np.argmin(dist)
cfmg9 = mg_9.Intensity['intensity'][:,idx]
#-----------------------------------------------------------------------------------------------------------------------
# Load the DEMs (manually cause the script is small and i'n not really into writing automated code today)
DEM_closed = np.array([2.79332238e+17,  1.00404790e+18,  2.96659525e+18,  6.56713288e+18,
  1.12843108e+19,  2.33374421e+19,  7.95381664e+19,  3.54399993e+20,
  1.33598547e+21,  3.39118803e+21,  5.25938190e+21,  4.83531173e+21,
  2.26927230e+21,  3.42013943e+20,  8.47334049e+21,  1.06107685e+22,
  4.31415279e+21,  8.95302327e+20,  1.38930100e+20,  2.61274111e+19,
  3.49522101e+18, 2.56128000e+19,  3.04127590e+20,  8.09440603e+20,
  8.33326943e+20,  2.19841351e+20, 0.0, 0.0, 0.0])

DEM_open = np.array([8.52272164e+16,  3.27356015e+17,  1.06929050e+18,  2.66802421e+18,
  5.24425608e+18,  1.07412325e+19,  3.36866801e+19,  1.55263093e+20,
  6.16576725e+20,  1.65376786e+21,  2.81081920e+21,  2.99850345e+21,
  2.11874559e+21,  1.74192321e+21,  3.90068959e+21,  3.45424455e+21,
  1.22733260e+21,  2.48743577e+20,  4.53694497e+19,  1.29050415e+19,
  7.19070286e+18,  1.52328308e+19,  6.09413360e+19,  1.34217014e+20,
  1.34906088e+20,  4.58799981e+19,  1.07767496e+18, 0.0, 0.0])
#-----------------------------------------------------------------------------------------------------------------------
# Function that integrates
def integral(x, func):
    int_value = 0
    for i in range(1,len(func)):
        int_value = int_value+(func[i-1]+func[i])*(x[i]-x[i-1])/2
    return  int_value
#-----------------------------------------------------------------------------------------------------------------------
# Set the intensities (manually to save run time)
int_ne_8_closed = 70.87749256426062
int_ne_8_open = 29.586777135215986
err_ne8 = 3.2186611955595916

int_mg_9_closed = 10.39440606907451
int_mg_9_open = 8.214783470553055
err_mg9 = 0.8819264438233863

int_ne_av = (int_ne_8_closed+int_ne_8_open)/2
int_mg_av =  (int_mg_9_closed+int_mg_9_open)/2
#-----------------------------------------------------------------------------------------------------------------------
# Final Boss
fip_bias_closed = (int_mg_9_closed/int_ne_8_closed)*(integral(temp, DEM_closed*cfne8)/integral(temp, DEM_closed*cfmg9))
fip_bias_open = (int_mg_9_open/int_ne_8_open)*(integral(temp, DEM_open*cfne8)/integral(temp, DEM_open*cfmg9))

fip_bias_2lr_closed = (int_mg_9_closed/int_ne_8_closed)*(np.max(cfne8)/np.max(cfmg9))
fip_bias_2lr_open = (int_mg_9_open/int_ne_8_open)*(np.max(cfne8)/np.max(cfmg9))

fip_bias_err = np.sqrt((err_ne8/int_ne_av)**2+(int_mg_av*err_mg9/int_ne_av**2)**2)/2
print(f'fip_err={fip_bias_err}')

fig = plt.figure(figsize=(16,9))
plt.subplot(311)
plt.plot(temp, DEM_closed*cfmg9, label = 'Mg9 closed')
plt.plot(temp, DEM_closed*cfne8, label = 'Ne8 clsoed')
plt.plot(temp, DEM_open*cfmg9, label = 'Mg9 open')
plt.plot(temp, DEM_open*cfne8, label = 'Ne8 open')
#plt.xlabel('Temperature (K)')
plt.ylabel('$G(n,T)  \tDEM(T)$ ($erg \t s^{-1} \t str^{-1} \t K^{-1}$)')
plt.title('$G(n,T) \t DEM(T)$')
plt.tight_layout()
plt.grid()
plt.xscale('log')
#plt.yscale('log')
plt.legend()

plt.subplot(312)
plt.plot(temp, cfne8,  label='$G(n,T) Ne_{VIII}$')
plt.plot(temp, cfmg9,  label='$G(n,T) Mg_{IX}$')
#plt.xlabel('Temperature (K)')
plt.ylabel('$G(n,T)$ ($erg \t s^{-1} \t str^{-1} \t cm^{5}$)')
plt.tight_layout()
plt.title('$G(n,T)$')
plt.grid()
plt.xscale('log')
#plt.yscale('log')
plt.legend()

plt.subplot(313)
plt.plot(temp, DEM_closed, label='DEM(T) closed')
plt.plot(temp, DEM_open,  label='DEM(T) open')
plt.xlabel('Temperature (K)')
plt.ylabel('$DEM(T)$ ($cm^{-5} \t K^{-1}$)')
plt.title('$DEM(T)$')
plt.tight_layout()
plt.grid()
plt.xscale('log')
#plt.yscale('log')
plt.legend()

print(f'FIP Bias closed: {fip_bias_closed}')
print(f'FIP Bias open: {fip_bias_open}')
print(f'FIP Bias 2LR closed: {fip_bias_2lr_closed}')
print(f'FIP Bias 2LR open: {fip_bias_2lr_open}')
#-----------------------------------------------------------------------------------------------------------------------
# FIP Bias Map because why not?

# Load the FITS files as sunpy maps and extract the data
import sunpy.map
ne_8_map = sunpy.map.Map('image_fit_ne_viii.fits').data
mg_9_map = sunpy.map.Map('image_fit_mg_ix.fits').data

# Avoid devision by zero
ne_8_map[ne_8_map == 0] = 1e-6
mg_9_map[mg_9_map == 0] = 1e-6

fip_bias_map = np.empty_like(ne_8_map)
fip_bias_2lr_map = np.empty_like(ne_8_map)
print(ne_8_map.shape, mg_9_map.shape)
for i in range(ne_8_map.shape[0]):
    for j in range(ne_8_map.shape[1]):
        fip_bias_map[i,j] = (mg_9_map[i,j]/ne_8_map[i,j])*(integral(temp, 0.5*(DEM_closed+DEM_open)*cfne8)/integral(temp, 0.5*(DEM_closed+DEM_open)*cfmg9))
        fip_bias_2lr_map[i,j] = (mg_9_map[i,j]/ne_8_map[i,j])*(np.max(cfne8) / np.max(cfmg9))
print((integral(temp, 0.5*(DEM_closed+DEM_open)*cfne8)/integral(temp, 0.5*(DEM_closed+DEM_open)*cfmg9)))
print(np.max(cfne8) / np.max(cfmg9))
# Plot the FIP Bias map
fig = plt.figure(figsize=(16,9))
plt.subplot(221)
plt.imshow(fip_bias_map, aspect='auto', cmap='viridis_r', origin='lower', vmin=0, vmax=6)
#contour_levels = np.linspace(0.99, 1.11, 2)  # Adjust number of levels
#plt.contour(fip_bias_map, levels=contour_levels, colors='r', linewidths=1)
plt.title('FIP Bias Map $Mg_{IX}/Ne_{VIII}$ DEM Method')
#plt.ylim(400,700)
#plt.xlim(75)
plt.colorbar(label='FIP Bias')

plt.subplot(222)
plt.imshow(fip_bias_2lr_map, aspect='auto', cmap='viridis_r', origin='lower', vmin=0, vmax=6)
#contour_levels = np.linspace(0.99, 1.11, 2)  # Adjust number of levels
#plt.contour(fip_bias_map, levels=contour_levels, colors='r', linewidths=1)
plt.title('FIP Bias Map $Mg_{IX}/Ne_{VIII}$ Two Line Ratio Method')
plt.colorbar(label='FIP Bias')
#plt.ylim(400,700)
#plt.xlim(75)

plt.subplot(223)
plt.imshow(ne_8_map, aspect='auto', vmin=np.percentile(ne_8_map,1), vmax=np.percentile(ne_8_map,99), origin='lower', cmap='viridis_r')
plt.colorbar(label='$W/m^2$')
plt.tight_layout()
plt.title('$Ne_{VIII}$')
#plt.ylim(400,700)
#plt.xlim(75)

plt.subplot(224)
plt.imshow(mg_9_map, aspect='auto', vmin=0, vmax=np.percentile(mg_9_map,98), origin='lower', cmap='viridis_r')
plt.colorbar(label='$W/m^2$')
plt.tight_layout()
plt.title('$Mg_{IX}$')
#plt.ylim(400,700)
#plt.xlim(75)

dem_cf_ratio = (integral(temp, cfmg9*DEM_closed)-integral(temp, cfmg9*DEM_open))/integral(temp, cfmg9*DEM_open)
print(f'MG IX: (integral(temp, cfmg9*DEM_closed)-integral(temp, cfmg9*DEM_open))/integral(temp, cfmg9*DEM_open)={dem_cf_ratio}')
dem_cf_ratio = (integral(temp, cfne8*DEM_closed)-integral(temp, cfne8*DEM_open))/integral(temp, cfne8*DEM_open)
print(f'Ne VIII: (integral(temp, cfne8*DEM_closed)-integral(temp, cfne8*DEM_open))/integral(temp, cfne8*DEM_open)={dem_cf_ratio}')
plt.show()