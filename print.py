# вроде должно работать

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm
import pylab as py
from scipy.fftpack import fft, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hls_to_rgb
import math

# const
lamb = 0.000532
k = (2 * np.pi) / lamb
alpha = 1e-4
gamma = 1
m = 1


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    A = (A + 0.1) % 1.0
    B = 1.0 - 1.0 / (1.0 + abs(z[idx]) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


# xi,yi = вектор
xi = yi = list(np.linspace(-1, 1, 100))

# x,y = матрица
x, y = np.meshgrid(xi, yi)


def r(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def fi(xi, yi):
    return ([[math.atan2(y[i][j], x[i][j]) for i in np.arange(0, len(xi))] for j in np.arange(0, len(yi))])


def f1(x, y):
    return np.sin(alpha * np.power(k * r(x, y), gamma) * m * fi(xi, yi))


z = (f1(x, y))
# print(z)

# we = ([[f1(x, y) for j in x] for i in y])
# print(we)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
ax.set_zticks([])
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

plt.imshow(z)
plt.show()

# o = np.abs(z)
# plt.imshow(o, origin='lower', interpolation='None')
# plt.show()

#
# ft = np.fft.fft2(z)
# ft = np.fft.fftshift(ft)
# # ft = np.abs(ft)
# # ft = np.log(ft)
#
#
#
# # plt.colorbar()
# img = colorize(ft)
# plt.imshow(img)
# #
# plt.show()
