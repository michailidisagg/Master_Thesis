import numpy as np

def equatorial_projection(R1, R2, phi):
    omega = np.arctan(np.sin(phi) / (-np.cos(phi) + (R2 / R1)))
    print('omega = ', np.degrees(omega))
    return R1 / np.sin(omega), omega

def eq_to_sky(l, theta):
    return l / np.cos(theta)

def compute_speed(L1, L2, delta_t):
    return (L2 - L1) / delta_t



def sigma_omega(R1, R2, phi, sigma_R1, sigma_R2, sigma_phi):
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    B = R1 / R2 - cos_phi
    f = sin_phi / B
    denom_common = (1 + f**2) * B**2

    d_omega_dR1 = -sin_phi / (denom_common * R2)
    d_omega_dR2 = sin_phi * R1 / (denom_common * R2**2)
    d_omega_dphi = ((B * cos_phi) - sin_phi**2) / denom_common

    sigma_omega_squared = (
        (d_omega_dR1 * sigma_R1)**2 +
        (d_omega_dR2 * sigma_R2)**2 +
        (d_omega_dphi * sigma_phi)**2
    )

    return np.sqrt(sigma_omega_squared)


def sigmaL(R1, omega, sigmaR1, sigma_omega):
    return np.sqrt((1/np.sin(omega))**2*sigmaR1**2+R1*(1/np.tan(omega))**2*sigma_omega**2)


def sigma_u(L1, L2, Dt, sigma_L1, sigma_L2, sigma_Dt):
    term1 = (sigma_L1 / Dt)**2
    term2 = (sigma_L2 / Dt)**2
    term3 = (((L2 - L1) / Dt**2) * sigma_Dt)**2
    return np.sqrt(term1 + term2 + term3)


# Inputs
theta = np.radians(55)
R_soho = [4.5*np.cos(theta), 5.5*np.cos(theta)]      # in solar radii
R_stereo = [6.5*np.cos(theta), 8.5*np.cos(theta)]  # in solar radii
sigma_R=0.5
sigma_Dt=300.0
sigma_phi= np.radians(0.5)
theta = np.radians(45)
R_soho = [4.5*np.cos(theta), 5.5*np.cos(theta)]      # in solar radii
R_stereo = [6.5*np.cos(theta), 8.5*np.cos(theta)]  # in solar radii
print(R_soho, R_stereo, theta)
delta_t = 1800.0         # in seconds
phi = np.radians(34.0)    # separation between spacecraft
  # polar angle of jet (from LOS)

# Projected length at first time
l1, omega1 = equatorial_projection(R_soho[0], R_stereo[0], phi)
# Projected length at second time
l2, omega2 = equatorial_projection(R_soho[1], R_stereo[1], phi)

# Deprojected (true) lengths
L1 = eq_to_sky(l1, theta)
L2 = eq_to_sky(l2, theta)
# Speed in solar radii per second
v = compute_speed(L1, L2, delta_t)

r_sun = 7.0e5
v_km_s = v * r_sun  # solar radius in km

sigma_omega = (omega1-omega2)/2
print(f'omega = {omega1}, {omega2}, {sigma_omega}')
sigma_L1 = sigmaL(R_soho[0], omega1, sigma_R, sigma_omega)
sigma_L2 = sigmaL(R_soho[1], omega1, sigma_R, sigma_omega)
sigma_u = sigma_u(L1,L2, delta_t, sigma_L1, sigma_L2, sigma_Dt)




print(f"Jet speed = {v_km_s:.2f} km/s")
print(f'sigma omega = {np.degrees(sigma_omega)} degrees')
print(f'sigma u = {sigma_u*r_sun} km/s')











#skecth of the geometry of the problem

#On making the image, initialy I put the names of the variables wrong, but then I kept it because it would be more confusing to me to change it again
#I have made sure the names that appear on the image are the correct ones
import matplotlib.pyplot as plt
import numpy as np

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.subplots_adjust(hspace=0.5)

# ---------------------------------------
# Top Panel: Plane of the Sky (Solar View)
# ---------------------------------------
ax = axes[0]
ax.set_title("Plane of the Sky")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Sun
sun = plt.Circle((0, 0), 0.2, color='gold')
ax.add_patch(sun)
ax.text(0, 0.25, 'Sun', ha='center')

# Jet heading southeast (positive x, negative y)
jet_x = 1.0
jet_y = -1.0

lx = 1.0
ly = 0.0
ax.arrow(0,0,lx,ly,  head_width=0.05, head_length=0.1, fc='blue', ec='blue', lw=2)
ax.text(0.3, 0.15, 'Jet (equatorial projection l)', color='blue')
ax.arrow(0, 0, jet_x, jet_y, head_width=0.05, head_length=0.1, fc='blue', ec='blue', lw=2)
ax.text(jet_x * 0, jet_y * 0.6 - 0.1, 'Jet (L)', color='blue')
v1 = np.array([lx, ly])
v2 = np.array([jet_x, jet_y])
cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(cos_theta)

# Draw arc to represent angle
arc_radius = 0.5
arc_theta = np.linspace(0, -angle, 100)
arc_x = arc_radius * np.cos(arc_theta)
arc_y = arc_radius * np.sin(arc_theta)
ax.plot(arc_x, arc_y, color='green', lw=2)
ax.text(arc_radius * 0.8, -0.1, r'$\theta$', color='green')

# ---------------------------------------
# Bottom Panel: Equatorial Plane View
# ---------------------------------------
ax = axes[1]
ax.set_title("Equatorial Plane")
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.axis('off')

# Sun
ax.add_patch(plt.Circle((0, 0), 0.2, color='gold'))
ax.text(0, 0.3, 'Sun', ha='center')

# Jet projections l1 and l2 (not on LOS!)
l1 = 4.5
l2 = 6.5
lx=1.0
ly=3.5
ax.arrow(0, 0, lx, ly, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
ax.text(-0.5, 3.7, 'jet equatorial projection (l)', color='blue')

ax.arrow(lx, ly, 0, 0.2-ly, head_width=0.1, head_length=0.2, fc='green', ec='green', linestyle='dotted')
ax.text(lx,-0.5 , 'l2 (projection at STEREO LOS)', color='green')




# Spacecraft separation angle φ
phi_deg = 35
phi_rad = np.radians(phi_deg)

# SOHO and STEREO LOS (not aligned with projections)
r_los = 6
soho_pos = [r_los, 0]
stereo_pos = [r_los * np.cos(phi_rad), r_los * np.sin(phi_rad)]

# SOHO LOS
ax.plot([soho_pos[0], 0], [soho_pos[1], 0], 'k--')
ax.text(soho_pos[0]-0.2 , 0, 'STEREO LOS', va='bottom')

# STEREO LOS
ax.plot([stereo_pos[0], 0], [stereo_pos[1], 0], 'k--')
ax.text(stereo_pos[0] + 0.3, stereo_pos[1] + 0.1, 'SOHO LOS', va='bottom')


# Stereo LOS unit vector
stereo_vec = np.array([stereo_pos[0], stereo_pos[1]])
stereo_unit = stereo_vec / np.linalg.norm(stereo_vec)

# Point to project: (lx, ly)
point = np.array([lx, ly])

# Project point onto stereo LOS
proj_length = np.dot(point, stereo_unit)
proj_point = proj_length * stereo_unit  # this is the foot of the perpendicular

# Draw arrow from (lx, ly) to the foot of perpendicular
ax.arrow(lx, ly,
         proj_point[0] - lx,
         proj_point[1] - ly,
         head_width=0.1,
         head_length=0.2,
         fc='red', ec='red', linestyle='dotted', length_includes_head=True)

# Optional: mark foot of projection
ax.text(proj_point[0] +0.1, proj_point[1]-0.2, 'l1 (projection at SOHO LOS)', color='red')


# Annotate angle φ (between spacecraft)
arc_radius = 1.5
arc_phi = np.linspace(0, phi_rad, 100)
arc_x = arc_radius * np.cos(arc_phi)
arc_y = arc_radius * np.sin(arc_phi)
ax.plot(arc_x, arc_y, color='black')
ax.text(arc_radius * 0.8, arc_radius * 0.4, r'$\phi$', color='black')

# Vectors
v1 = np.array([lx, ly])
v2 = np.array([stereo_pos[0], stereo_pos[1]])

# Normalize
v1_u = v1 / np.linalg.norm(v1)
v2_u = v2 / np.linalg.norm(v2)

# Angle between them (omega)
dot = np.dot(v1_u, v2_u)
omega_rad = np.arccos(dot)

# Direction of rotation (sign of cross product z-component)
cross = np.cross(np.append(v1_u, 0), np.append(v2_u, 0))
sign = np.sign(cross[2])  # positive = CCW, negative = CW

# Generate arc
arc_radius = 1.2
arc_steps = 100
start_angle = np.arctan2(v1_u[1], v1_u[0])
arc_angles = np.linspace(start_angle, start_angle + sign * omega_rad, arc_steps)

arc_x = arc_radius * np.cos(arc_angles)
arc_y = arc_radius * np.sin(arc_angles)

# Plot arc and label
ax.plot(arc_x, arc_y, color='black', lw=2)
mid_angle = start_angle + sign * omega_rad / 2
ax.text(arc_radius * np.cos(mid_angle) + 0.1,
        arc_radius * np.sin(mid_angle) + 0.1,
        r'$\omega$', color='black')


plt.show()
