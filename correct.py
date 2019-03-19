from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hls_to_rgb


wl = 0.000532
k = (2 * np.pi) / wl
alpha = 1e-3
gamma = 1
m = 1


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)

# r = np.sqrt(np.power(x, 2) + np.power(y, 2))


def fi():
    for y in range(-1, 2):
        for x in range(-1, 2):
            return math.atan2(y, x)


# fi = math.atan2(y, x)


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
#
# def f(r, fi):
    # return np.sin(alpha * np.float_power(k * r, gamma) * m * fi)

def f1(x, y):
    return np.sin(alpha * np.float_power(k * np.sqrt((x ** 2 + y ** 2)), gamma) * m)




z = f1(x, y)


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.plot_surface(x, y, z)
# ax.set_zticks([])
# # убрать коммент. если надо убрать значения оси z
# ax.set_xticks([])
# ax.set_yticks([])
#
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
#
plt.imshow(z, origin='lower', interpolation='None')
plt.show()

# plt.show()
# o = np.abs(z)
# plt.imshow(o, origin='lower', interpolation='None')
# plt.show()

# ft = np.fft.fft2(z)
# ft = np.fft.fftshift(ft)
# ft = np.abs(ft)
# ft = np.log(ft)
#
# img = colorize(f1(x, y))
# plt.imshow(img)
#
# plt.show()
#
