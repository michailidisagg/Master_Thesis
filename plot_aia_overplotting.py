import os
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
import sunpy.sun.constants
import pfsspy
import pfsspy.tracing as tracing
from sunpy.coordinates import frames
import astropy.constants as const
#-----------------------------------------------------------------------------------------------------------------------
#maps

#metis map 1
filename1 = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\metis_workshop\\2022_03_03\\solo_L2_metis-vl-pb_20220303T140501_V01.fits"
hdulist = fits.open(filename1)
hdulist.info()
data = hdulist[0].data
header = hdulist[0].header
metis_map1 = sunpy.map.Map(data, header)
hdulist.close()
#metis map 2
filename2 = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\metis_workshop\\2022_03_03\\solo_L2_metis-vl-pb_20220303T220501_V01.fits"
hdulist = fits.open(filename2)
hdulist.info()
data = hdulist[0].data
header2 = hdulist[0].header
metis_map2 = sunpy.map.Map(data, header)
hdulist.close()
#difference metis map
metis_map = sunpy.map.Map(metis_map1.data - metis_map2.data, metis_map1.meta)
#metis_map = sunpy.map.Map("C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\metis_workshop\\2022_03_03\\cor1\\COR1.jp2")
#gong map
gong_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\gong_2022_03_07.fits"
gong_map = sunpy.map.Map(gong_fname)
gong_map.meta['cunit1'] = 'deg'
gong_map.meta['cunit2'] = 'deg'

#aia map
aia_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\aia_lev1_193a_2022_03_03t14_19_04_83z_image_lev1.fits"
aia = sunpy.map.Map(aia_fname)

#hmi maps
hmi_before_jet_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\hmi_m_45s_2022_03_03_14_15_45_tai_magnetogram.fits"
hmi_after_jet_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\hmi_m_45s_2022_03_03_14_20_15_tai_magnetogram.fits"
hmi_synoptic = sunpy.map.Map("C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\hmi.Synoptic_Mr.2254.fits")
hmi_synoptic_data = np.nan_to_num(hmi_synoptic.data, nan=0.0)
hmi_synoptic = sunpy.map.Map(hmi_synoptic_data, hmi_synoptic.meta).resample([360,180]*u.pix)
hmi_720_br_map = sunpy.map.Map("C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\br_720_map.fits").reproject_to(gong_map.wcs)
hmi_720_br_map_data = np.nan_to_num(hmi_720_br_map.data, nan=0.0)
hmi_720_br_map = sunpy.map.Map(hmi_720_br_map_data, hmi_720_br_map.meta)
hmi_before_map = sunpy.map.Map(hmi_before_jet_fname).resample([360, 180] * u.pix).reproject_to(gong_map.wcs)
hmi_after_map = sunpy.map.Map(hmi_after_jet_fname).resample([360, 180] * u.pix).reproject_to(gong_map.wcs)
hmi_before_data = np.nan_to_num(hmi_before_map.data, nan=0.0)
hmi_after_data = np.nan_to_num(hmi_after_map.data, nan=0.0)
hmi_before_map = sunpy.map.Map(hmi_before_data, hmi_before_map.meta)
hmi_after_map = sunpy.map.Map(hmi_after_data, hmi_after_map.meta)


#eui maps
eui_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\solo_L1_eui-fsi174-image_20220303T141045183_V02.fits"
eui_map = sunpy.map.Map(eui_fname)
#euvi maps
euvi_fname = "C:\\Users\\ageli\\Desktop\\uoa\\Master_thesis_TATATATA!!!\\Master_thesis_py\\20220303_141500_n4eua.fts"
euvi_map = sunpy.map.Map(euvi_fname)

maps = [aia, eui_map, euvi_map]
#-----------------------------------------------------------------------------------------------------------------------
#When using an hmi map that only covers one hemisphere, this will full the other hemisphere with the opposite values,
#ensuring the total flux is zero
def adjust_flux_to_zero(magnetogram_map):
    # Convert NaN values (back hemisphere) to zero
    front_hemisphere_data = magnetogram_map.data
    front_hemisphere_data[np.isnan(front_hemisphere_data)] = 0.0

    # Copy the original data array and flip the copy
    flipped_data = np.copy(front_hemisphere_data)
    flipped_data = np.flip(flipped_data, axis=1)  # Flip the data horizontally (back hemisphere)

    # New data: front hemisphere unchanged, back hemisphere is opposite
    new_data = front_hemisphere_data - flipped_data

    # Create a new map with adjusted data
    adjusted_map = sunpy.map.Map(new_data, magnetogram_map.meta)

    return adjusted_map
#hmi_720_br_map_adj = adjust_flux_to_zero(hmi_720_br_map)
#-----------------------------------------------------------------------------------------------------------------------
#run pfss
def pfss_run(magnetic_fild_, nrho, rss, lon_min, lon_max, lat_min, lat_max, num_lines_lon, num_lines_lat, projection):
    pfss_in = pfsspy.Input(magnetic_fild_, nrho, rss)
    hp_lon = np.linspace(lon_min, lon_max, num_lines_lon) * u.arcsec
    hp_lat = np.linspace(lat_min, lat_max, num_lines_lat) * u.arcsec
    lon, lat = np.meshgrid(hp_lon, hp_lat)
    seeds = SkyCoord(lon.ravel(), lat.ravel(), frame=projection.coordinate_frame)
    pfss_out = pfsspy.pfss(pfss_in)
    tracer = tracing.FortranTracer()
    flines = tracer.trace(seeds, pfss_out)
    return flines

#flines_before = pfss_run(hmi_before_map, 100, 2.5, -550, 50, -550, 50, 10, 10, aia)
#flines_after = pfss_run(hmi_after_map, 100, 2.5, -550, 50, -550, 50, 10, 10, aia)
flines_720_br = pfss_run(hmi_720_br_map, 100, 2.5, -500,100, -500, 100,30, 30, aia)
#-----------------------------------------------------------------------------------------------------------------------
#plot pfss results
def plot(maps, flines_before, flines_after):
    levels = np.array([-100, 100]),
    for map in maps:
        fig = plt.figure()
        plt.suptitle('PFSS 2022-03-03')
        ax1 = plt.subplot(1, 2, 1, projection=map)
        map.plot(axes=ax1)
        plt.title(f'Before jet, projection: {map.instrument}')
        contour = ax1.contour(hmi_before_map.data, levels=levels, colors=['blue', 'red'],
                              transform=ax1.get_transform(hmi_before_map.wcs), alpha=0.5)
        for fline in flines_before:
            color = {0: 'gray', -1: 'blue', 1: 'red'}.get(fline.polarity)
            ax1.plot_coord(fline.coords, linewidth=1.0, color=color)

        ax2 = plt.subplot(1, 2, 2, projection=map)
        map.plot(axes=ax2)
        plt.title(f'During jet, projection: {map.instrument}')
        contour = ax2.contour(hmi_after_map.data, levels=levels, colors=['blue', 'red'],
                              transform=ax2.get_transform(hmi_after_map.wcs), alpha=0.5)
        for fline in flines_after:
            color = {0: 'gray', -1: 'blue', 1: 'red'}.get(fline.polarity)
            ax2.plot_coord(fline.coords, linewidth=1.0, color=color)
    plt.tight_layout
    plt.show()

def plot_single_map(maps, flines):
    levels = np.array([-100, 100])
    for map in maps:
        fig = plt.figure()
        plt.suptitle('PFSS 2022-03-03')
        ax1 = plt.subplot(1, 1, 1, projection=map)
        map.plot(axes=ax1)
        plt.title( f'Projection: {map.instrument}')
        #contour = ax1.contour(hmi_720_br_map.data, levels=levels, colors=['blue', 'red'],
         #                     transform=ax1.get_transform(hmi_720_br_map.wcs), alpha=0.5)
        for fline in flines:
            color = {0: 'white', -1: '#00bfff', 1: 'red'}.get(fline.polarity)
            ax1.plot_coord(fline.coords, linewidth=1.0, color=color, alpha=0.5)
    plt.tight_layout
    plt.show()
plot_single_map(maps, flines_720_br)
#get B vector at sun surface
def get_sun_surface_vector_b(pfss_out):
    bg = pfss_out.bg
    # Surface of the Sun is at radial index 0
    br = bg[:, :, 0, 2]  # Radial component
    btheta = bg[:, :, 0, 1]  # Theta (latitudinal) component
    bphi = bg[:, :, 0, 0]  # Phi (longitudinal) component
    return br, btheta, bphi


#-----------------------------------------------------------------------------------------------------------------------
#run againe for full disk
#flines_before = pfss_run(hmi_before_map, 100, 2.5, -680, 680, -680, 680, 14, 14, aia)
#flines_after = pfss_run(hmi_after_map, 100, 2.5, -680, 680, -680, 680, 14, 14, aia)
#-----------------------------------------------------------------------------------------------------------------------
#overplot on metis
def plot_metis(flines_before):
    fig = plt.figure()
    #plt.suptitle('PFSS 2022-03-03')
    ax1 = plt.subplot(1, 1, 1, projection=metis_map)
    vmin = np.percentile(metis_map.data, 1)
    vmax = np.percentile(metis_map.data, 99)
    #metis_map1.plot(axes=ax1, vmin=vmin, vmax=vmax)
    metis_map1.plot(axes=ax1, vmin=0, vmax=1e-9)
    metis_map1.draw_limb()
    metis_map1.draw_grid()
    #eui_map.plot(axes=ax1, autoalign=True)
    #plt.title(f'Projection: {metis_map.instrument} and {eui_map.instrument}, {metis_map1.date} ')
    #levels = np.array([-300, 300])
    #contour = ax1.contour(hmi_before_map.data, levels=levels, colors=['blue', 'red'],
                          #transform=ax1.get_transform(hmi_before_map.wcs), alpha=0.5)
    for fline in flines_before:
        color = {0: 'white', -1: 'blue', 1: 'red'}.get(fline.polarity)
        ax1.plot_coord(fline.coords, alpha=0.5, linewidth=0.5, color=color)

    plt.show()
#plot_metis(flines_720_br)


