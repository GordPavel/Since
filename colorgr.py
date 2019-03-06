import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hls_to_rgb
from matplotlib import axes

def colorize(z):
    n, l = z.shape
    c = np.zeros((n, l, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    A = (A + 0.1) % 1.0
    B = 1.0 - 1.0 / (1.0 + abs(z[idx]) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c
# x, y = np.mgrid[-200:200, -200:200]
# dist = np.hypot(x, y) # Linear distance from point 0, 0

x = np.linspace(-1, 1)
y = np.linspace(-1, 1)

x, y = np.meshgrid(x, y)

lamb = 0.000532
k = 2*np.pi/lamb
gamma = 1
alpha = 0.001
r = np.sqrt(np.power(x, 2) + np.power(y, 2))
m = 1

N = 1
Rmax = 1
h_x = 2*Rmax/N

# for n in range(1, N):
#     xx[n] = -Rmax + n * h_x
# for m in range(1, N):
#     yy[m] = -Rmax + m *h_y
#
# r[n, m] =

def fi():
    for n in range(-1, 1):
        for t in range(-1, 1):
            return math.atan2(n, t)

z = np.sin(alpha * np.float_power(k * r, gamma)* m * fi())

for u in range(-1, 1):
    print(fi())



fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(x, y, z)
ax.set_zticks([])
# # убрать коммент. если надо убрать значения оси z
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# plt.imshow(z, origin='lower', interpolation='None')
# plt.show()
#
# # plt.show()
# o = np.abs(z)
# plt.imshow(o, origin='lower', interpolation='None')
# # plt.show()
#
# ft = np.fft.fft2(z)
# # ft = np.fft.fftshift(ft)
# # ft = np.abs(ft)
# # ft = np.log(ft)
#
# # img = colorize(ft)
# # plt.imshow(img)
#
# plt.show()