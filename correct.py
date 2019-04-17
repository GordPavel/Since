from matplotlib import pyplot as plt
import numpy as np
import math
from colorsys import hls_to_rgb


wl = 0.000532
k = (2 * np.pi) / wl
alpha = 1e-3
# angle
gamma = 1
m = 1

# making 2-D dimensional space
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)


def fi():
    for y_val in range(-1, 2):
        for x_val in range(-1, 2):
            return math.atan2(y_val, x_val)


def colorize( values ):
    n, l = values.shape
    c = np.zeros((n, l, 3))
    c[np.isinf( values ) ] = (1.0, 1.0, 1.0)
    c[np.isnan( values ) ] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf( values ) + np.isnan( values ))
    a = (np.angle( values[idx ] ) + np.pi) / (2 * np.pi)
    a = (a + 0.1) % 1.0
    b = 1.0 - 1.0 / (1.0 + abs( values[idx ] ) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(a, b)]
    return c


def f1( x_val , y_val ):
    return np.sin( alpha * np.float_power( k * np.sqrt( (x_val ** 2 + y_val ** 2) ) , gamma ) * m )


z = f1(x, y)
plt.imshow(z, origin='lower', interpolation='None')
plt.show()
