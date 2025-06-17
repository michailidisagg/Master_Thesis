"""
The code performs a linear force free magnetic field extrapolation at planar geometry.
It is based on JR Costa's idl code.
Input: Bz(x,y,z=0)
Output: Bx(x,y,z), By(x,y,z), Bz(x,y,z)
Author: Angelos Michailidis
Afilliation: National and Kapodistrian University of Athens
"""
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#Import necessery modules
import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
import sunpy.sun.constants
from sunpy.coordinates import frames
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from mayavi import mlab
from scipy.integrate import odeint
# ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#Define the functions
#rebin data
def rebin(BLOS ,nx, ny, rebin_factor):
    if nx % rebin_factor != 0 or ny % rebin_factor != 0:
        # Adjust dimensions to nearest divisible sizes
        nx = nx - (nx % rebin_factor)
        ny = ny - (ny % rebin_factor)
        BLOS = BLOS[:nx, :ny]
    # Reshape and rebin
    BLOS = BLOS.reshape(nx // rebin_factor, rebin_factor, ny // rebin_factor, rebin_factor).mean(axis=(1, 3))
    # Update the shape
    nx, ny = BLOS.shape
    print('rebin completed')
    return BLOS, nx, ny
#-----------------------------------------------------------------------------------------------------------------------
#The function that performs the extrapolation
def lff(bz0, z, a, nx1, ny1, nz):
    #ensure correct boundary conditions
    b00 = 0.0 #if some assumptions are valid, change that to b00=bz0.mean() (see JR Costa's code)
    bz0e = bz0
    seehafer = 0 # set this to 0, if some assumptions are valid (see JR Costa's code)
    if seehafer == 0:
       bz0e[0:nx1, 0:ny1] = bz0 - b00
       bz0e[nx1:nx, 0:ny1] = -np.rot90(bz0, k=5)[:nx-nx1, :ny1]
       bz0e[0:nx1, ny1:ny] = -np.rot90(bz0, k=7)[:nx1, :ny-ny1]
       bz0e[nx1:nx, ny1:ny] = -np.rot90(bz0e[0:nx1, ny1:ny], k=5)[:nx-nx1, :ny-ny1]
    else:
        bz0e = bz0-b00
    #create wavenumbers
    # Be careful on how python and idl handle the kx and ky arrays. If not careful, kx and ky might be inversed
    kx = 2 * np.pi * np.fft.fftfreq(nx1)
    ky = 2 * np.pi * np.fft.fftfreq(ny1)
    ky, kx = np.meshgrid(kx, ky, indexing='ij')
    #FFT of input B (with the correct boundary conditions)
    fbz0 = np.fft.fft2(bz0e)
    #Compute the wave vectors
    kz = np.sqrt(np.maximum(kx ** 2 + ky ** 2 - a ** 2, 0)) #keep only small scale solutions and only positive k
    kz = np.nan_to_num(kz,nan=1e-10) #avoid numerical error with the large scale solutions (make them zero)
    kz[kz==0]=1e-10 #this is to avoid devision by zero later
    #Compute Green functions in Fourier space and applly convolution theorem
    #the sign is positive (instead of negative) to deal with the way python handles the kx,ky arrays
    argx = fbz0 * (1j) * (kx * kz - a * ky) / (kz ** 2 + a ** 2)
    argy = fbz0 * (1j) * (ky * kz + a * kx) / (kz ** 2 + a ** 2)
    #initalize np arrays to store the results
    bx = np.empty((nx, ny, nz))
    by = np.empty((nx, ny, nz))
    bz = np.empty((nx, ny, nz))
    #Apply inverse FT to get the solution. Loop in Z levels
    for j in range(nz):
        print(f'computing level {j}/{nz-1}') #monitor the progress of the computation
        exp_term = np.exp(-kz * z[j])
        bx[:, :, j] = np.real(np.fft.ifft2(argx * exp_term))[:nx1, :ny1]
        by[:, :, j] = np.real(np.fft.ifft2(argy * exp_term))[:nx1, :ny1]
        bz[:, :, j] = np.real(np.fft.ifft2(fbz0 * exp_term))[:nx1, :ny1] + b00
    print('Computation of extrapolation completed')
    return bx, by, bz
# -----------------------------------------------------------------------------------------------------------------------
#quick visualization to check if the extrapolation worked (the user may substitute any level bi[x,y,z] they wish)
def quick_vis(Bx, By, Bz):
    fig = plt.figure()
    plt.subplot(231)
    plt.imshow(Bx[:, :, 0], cmap="gray")
    plt.title("Bx, a=0, z=0")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(232)
    plt.imshow(By[:, :, 0], cmap="gray")
    plt.title("By, a=0, z=0")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(233)
    plt.imshow(Bz[:, :, 0], cmap="gray")
    plt.title("Bz, a=0, z=0")
    plt.colorbar()
    plt.gca().invert_yaxis()

    plt.subplot(234)
    plt.imshow(Bx[:, :, 5], cmap="gray")
    plt.title("Bx, a=0, z=5")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(235)
    plt.imshow(By[:, :, 5], cmap="gray")
    plt.title("By, a=0, z=5")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(236)
    plt.imshow(Bz[:, :, 5], cmap="gray")
    plt.title("Bz, a=0, z=5")
    plt.colorbar()
    plt.gca().invert_yaxis()

    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#make it again sunpy map for solar physics applications (the user may substitute any level bi[x,y,z] they wish)
def np_to_sunpy(bx,by,bz,a,z):
    #plot bx,by,bz on photosphere
    mapx = sunpy.map.Map(bx[:, :, 0], BLOS1.meta)
    mapy = sunpy.map.Map(by[:, :, 0], BLOS1.meta)
    mapz = sunpy.map.Map(bz[:, :, 0], BLOS1.meta)
    fig = plt.figure()
    ax1 = fig.add_subplot(231, projection=mapx)
    mapx.plot(axes=ax1)
    plt.colorbar(label = 'Gauss')
    ax2 = fig.add_subplot(232, projection=mapy)
    mapy.plot(axes=ax2)
    plt.colorbar(label = 'Gauss')
    ax3 = fig.add_subplot(233, projection=mapz)
    mapz.plot(axes=ax3)
    plt.colorbar(label = 'Gauss')
    ax1.set_title(f'Bx, a={a} $pixel^{{-1}}$, z=0cm')
    ax2.set_title(f'By, a={a} $pixel^{{-1}}$, z=0cm')
    ax3.set_title(f'Bz, a={a} $pixel^{{-1}}$, z=0cm')

    #plot bx,by,bz on a plane above photsphere
    #z values are in units of pixels. A pixel corresponds to 715e5cm
    mapx = sunpy.map.Map(bx[:, :, z], BLOS1.meta)
    mapy = sunpy.map.Map(by[:, :, z], BLOS1.meta)
    mapz = sunpy.map.Map(bz[:, :, z], BLOS1.meta)
    ax4 = fig.add_subplot(234, projection=mapx)
    mapx.plot(axes=ax4)
    plt.colorbar(label = 'Gauss')
    ax5 = fig.add_subplot(235, projection=mapy)
    mapy.plot(axes=ax5)
    plt.colorbar(label = 'Gauss')
    ax6 = fig.add_subplot(236, projection=mapz)
    mapz.plot(axes=ax6)
    plt.colorbar(label = 'Gauss')
    ax4.set_title(f'Bx, a={a} $pixel^{{-1}}$, z={z*715}km')
    ax5.set_title(f'By, a={a} $pixel^{{-1}}$, z={z*715}km')
    ax6.set_title(f'Bz, a={a} $pixel^{{-1}}$, z={z*715}km')

    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#draw 3d field lines
#Part of this code was made using chat gtp. I inserted as promt the numerical mathematical approximation for tracing
#field lines and it wrote this function to compute the lines
#In the end I didn't use this in my thesis, but if run properly, it works
def visualize_magnetic_field(bx, by, bz, num_seed_points_x, num_seed_points_y, ds, max_steps, a):
    # Define box dimensions from input field
    nx, ny, nz = bx.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    # Generate seed points directly from the existing grid
    seed_x = np.linspace(0, nx - 1, num_seed_points_x, dtype=int)
    seed_y = np.linspace(0, ny - 1, num_seed_points_y, dtype=int)
    seed_points = [(sx, sy, 0) for sx in seed_x for sy in seed_y]
    # Create interpolators for each field component
    interp_bx = RegularGridInterpolator((x, y, z), bx)
    interp_by = RegularGridInterpolator((x, y, z), by)
    interp_bz = RegularGridInterpolator((x, y, z), bz)
    # Function to interpolate the magnetic field at a given position
    def interpolate_field(s):
        s = np.clip(s, [0, 0, 0], [nx - 1, ny - 1, nz - 1])
        b_x = interp_bx((s[0], s[1], s[2]))
        b_y = interp_by((s[0], s[1], s[2]))
        b_z = interp_bz((s[0], s[1], s[2]))
        b = np.array([b_x, b_y, b_z], dtype=float)
        return np.squeeze(b)
    # Trace a field line starting from a seed point
    def trace_field_line(s0):
        s = np.array(s0, dtype=float)
        path = [s.copy()]  # Start with the seed point
        for _ in range(max_steps):
            b = interpolate_field(s)
            b_norm = np.linalg.norm(b)
            if b_norm == 0:  # Avoid division by zero
                break
            b_unit = b / b_norm
            s = s + b_unit * ds
            if np.any(s < 0) or np.any(s >= [nx, ny, nz]):  # Check bounds
                break
            path.append(s.copy())
        return np.array(path, dtype=float)
    field_lines = [trace_field_line(s0) for s0 in seed_points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Y, X = np.meshgrid(x, y, indexing='ij')
    ax.contourf(X, Y, bz[:, :, 0], zdir='z', cmap='gray', alpha=0.1, offset=0, levels=500)
    for field_line in field_lines:
        if field_line[-1, 2] <= 1.0:
            ax.plot(field_line[:, 1], field_line[:, 0], field_line[:, 2], color='blue', linewidth=1)
        else:
            ax.plot(field_line[:, 1], field_line[:, 0], field_line[:, 2], color='red', linewidth=1)
    ax.set_title('HMI')
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_zlabel("z (pixels)")
    ax.plot([], [], color='blue', label='Closed lines')
    ax.plot([], [], color='red', label='Open lines')
    ax.legend()
    ax.set(zlim=(0, nz))
    plt.suptitle(f'Linear Force Free Extrapolation, $a={a} \\text{{pixel}}^{{-1}}$')

    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#trace interactive magnetic field lines with mayavi module
#This is the one I use in my thesis
def vis_mayavi(nx, ny, nz, bx, by, bz):
    X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nz:1]  # create a grid
    #X = X[:, :, ::-1]
    #Y = Y[:, :, ::-1]
    #Z = Z[:, :, ::-1]
    #Rotate because of the way numpy handles axes
    bx = bx[::-1, ::-1, :]
    by = by[::-1, ::-1, :]
    bz = bz[::-1, ::-1, :]

    mlab.figure(bgcolor=(0, 0, 0)) #load a mayavi figure
    streamlines = mlab.flow(X, Y, Z, bx.transpose(1, 0, 2), by.transpose(1, 0, 2), bz.transpose(1, 0, 2), seed_visible=True, seedtype='plane', seed_resolution=25) #compute the field lines
    # customize the appearence
    streamlines.streamline_type = 'tube' #customize the appearence
    streamlines.tube_filter.radius = 1.5
    streamlines.stream_tracer.integration_direction = 'both'
    #create the seed for the tracing
    streamlines.seed.widget.origin = (0, 0, 1)
    streamlines.seed.widget.point1 = (nx, 0, 1)
    streamlines.seed.widget.point2 = (0, ny, 1)
    streamlines.stream_tracer.maximum_propagation = max(nx, ny, nz)
    mlab.show()
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#Main program
# Load SunPy Map (the magnetogram)
BLOS = sunpy.map.Map("br_720_map.fits") #substitute your magnetogram's file path
"""In my version I pass as input B a Br magnetogram from HMI_720s vector field series. Using the approximation that a small
region of the sphere can be approximated as a plane, I assume Br=Bz. If you use this code with HMI_45s BLOS data, make sure
you have preprocessed the data to convert BLOS to BZ."""
#-----------------------------------------------------------------------------------------------------------------------
# Define submap coordinates (user may cropp the map in any coords to isolate AR of interest)
bottom_left = SkyCoord(-550 * u.arcsec, -550 * u.arcsec, frame=frames.Helioprojective, observer=BLOS.observer_coordinate)
top_right = SkyCoord(50 * u.arcsec, 50 * u.arcsec, frame=frames.Helioprojective, observer=BLOS.observer_coordinate)
# Create the submap
BLOS1 = BLOS.submap(bottom_left=bottom_left, top_right=top_right) #I keep this because I'll use its meta for plotting
BLOS = np.array(BLOS.submap(bottom_left=bottom_left, top_right=top_right).data)
nx, ny = BLOS.shape
#rebin to reduse resolution (to make computation faster for high z)
#comment this out if you want the original hmi resolution
BLOS, nx, ny = rebin(BLOS, nx, ny, 3)
#-----------------------------------------------------------------------------------------------------------------------
# Define parameters for calculation
a = 0. #a const in 1/pixel
nz = 200 #number of z levels in pixels
z = np.arange(nz)
#-----------------------------------------------------------------------------------------------------------------------
# Perform LFF extrapolation
bx, by, bz = lff(BLOS, z, a, nx, ny, nz)
#-----------------------------------------------------------------------------------------------------------------------
# quick vis to check if it worked
#quick_vis(bx, by, bz) #plots bx,by,bz magnetograms in z=0 and z=5 planes as np arrays

#proper vis for applications
#np_to_sunpy(bx,by,bz,a,z=5) #plots magnetograms of bx,by,bz in the xy planes for z=0, and for custom z as sunpy maps
#visualize_magnetic_field(bx, by, bz, num_seed_points_x=80, num_seed_points_y=80, ds=5, max_steps=5000, a=a) #traces magnetic field lines using matplotlib
vis_mayavi(nx,ny,nz,bx,by,bz) #traces magnetic field lines using mayavi (default, user may uncomment the other visualization functions (default))
