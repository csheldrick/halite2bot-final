import numpy as np
from scipy.interpolate import Rbf
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib
import time
from hlt.entity import Ship
from math import sqrt, exp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


class WeightedObject:
    def __init__(self, x, y, w, g):
        self.x = x
        self.y = y
        self.w = w
        self.g = g


def vector(x):
    return np.array([x.x, x.y])


def unit_vector(x):
    return x / np.linalg.norm(x)


def norm(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=0))


def euclidean_norm(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=0))


def gaussian(epsilon, r):
    return np.exp(-(1.0 / epsilon * r) ** 2)


def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return np.sqrt(distance2(x1, y1, x2, y2))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def _get_epsilon(*args):
    xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                     for a in args[:-1]])
    N = xi.shape[-1]
    dim = xi.shape[0]
    ximax = np.amax(xi, axis=1)
    ximin = np.amin(xi, axis=1)
    edges = ximax - ximin
    edges = edges[np.nonzero(edges)]
    epsilon = np.power(np.prod(edges) / N, 1.0 / edges.size)
    return epsilon


def get_epsilon(game_map, make=True):
    w = game_map.width
    h = game_map.height
    r = []
    me = game_map.get_me()
    fs = me.all_ships()
    gridX, gridY = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')

    fsX, fsY, fsZ, o = make_data(fs, cls='ship', nef='f', color='green')

    xs, ys, zs = np.array([]), np.array([]), np.array([])
    xs = np.concatenate([xs, fsX])
    ys = np.concatenate([ys, fsY])
    zs = np.concatenate([zs, fsZ])
    all_planets = game_map.all_planets()
    u_planets = [p for p in all_planets if not p.is_owned()]
    if u_planets:
        uX, uY, uZ, o = make_data(u_planets, cls='planet', nef='f', objs=o)
        xs = np.concatenate([xs, uX])
        ys = np.concatenate([ys, uY])
        zs = np.concatenate([zs, uZ])

    f_planets = [p for p in all_planets if p not in u_planets and p.owner == me]
    if f_planets:
        fX, fY, fZ, o = make_data(f_planets, cls='planet', nef='f', color='green', objs=o)
        xs = np.concatenate([xs, fX])
        ys = np.concatenate([ys, fY])
        zs = np.concatenate([zs, fZ])

    e_planets = [p for p in all_planets if p not in u_planets and p.owner != me]
    if e_planets:
        eX, eY, eZ, o = make_data(e_planets, cls='planet', nef='e', color='red', objs=o)
        xs = np.concatenate([xs, eX])
        ys = np.concatenate([ys, eY])
        zs = np.concatenate([zs, eZ])

    es = [s for s in game_map._all_ships() if s.owner != me]
    if es:
        esX, esY, esZ, o = make_data(es, cls='ship', nef='e', color='red', objs=o)
        xs = np.concatenate([xs, esX])
        ys = np.concatenate([ys, esY])
        zs = np.concatenate([zs, esZ])
    x, y, z = make_edges(gridX, gridY)
    xs = np.concatenate([xs, x])
    ys = np.concatenate([ys, y])
    zs = np.concatenate([zs, z])
    allx = xs
    ally = ys
    allz = zs
    eps = _get_epsilon(allx, ally, allz)
    r.append(eps)
    # r.append(allx)
    # r.append(ally)
    # r.append(allz)
    if make:
        F = make_rbf(allx, ally, allz, function='gaussian', smooth=-5, epsilon=17)  # smooth=-5 epsilon=17
        r.append(F)
        Z = make_z(F, gridX, gridY)
        r.append(Z)
    return r


def get_gradient(my_ships, all_objects, game_map):
    eps = get_epsilon(game_map, make=False)[0]
    my_id = my_ships[0].owner
    gradU, gradV = np.zeros(len(my_ships)), np.zeros(len(my_ships))
    for idx, s in enumerate(my_ships):
        x = int(s.x)
        y = int(s.y)
        sv = vector(s)
        gradX, gradY = 0, 0
        for _o in all_objects:
            if _o == s: continue
            ov = vector(_o)
            r = euclidean_norm(ov, sv) * _o.radius
            # print("r {}".format(r))
            o = w_g_o(_o, r, my_id, eps)
            # print("o {}".format(o))
            d = distance(x, y, o.x, o.y)
            # print("d {}".format(d))
            f = o.w * exp(-o.g * d)
            # print("f {}".format(f))
            gradX += o.w * o.g * f * (x - o.x)
            gradY += o.w * o.g * f * (y - o.y)
        gradU[idx] = gradX
        gradV[idx] = gradY
    return gradU, gradV


def weight(t, cls='ship', nef='f'):
    if cls is 'ship':
        if nef is 'f': return 1.
        if nef is 'e': return -1.
    elif cls is 'planet':
        if nef is 'u' or nef is 'n':
            return -1.
        elif nef is 'f':
            return -1.
        elif nef is 'e':
            return -1.


def get_radius(p, r=4):
    r = int(p.radius) + r
    n = []
    for i in range(-r, r):
        n.append([p.x + i, p.y])
        n.append([p.x, p.y + i])
    return n


def make_edges(x, y):
    xs, ys, zs = [], [], []
    edge = 1.0
    for i in range(len(x[0])):  # All cols
        xs.append(0)  # first row
        ys.append(i)  # each col
        zs.append(edge)  # edge value
        xs.append(len(x) - 1)  # Last row
        ys.append(i)  # each  col
        zs.append(edge)  # edge value
    for i in range(len(x)):  # All rows
        xs.append(i)  # each row
        ys.append(0)  # first col
        zs.append(edge)  # edge value
        xs.append(i)  # each row
        ys.append(len(x[0]) - 1)  # Last col
        zs.append(edge)
    return xs, ys, zs


def make_corners(x, y):
    xs, ys, zs = [], [], []
    edge = 1.0
    xs.append(0)
    ys.append(0)
    zs.append(edge)
    xs.append(0)
    ys.append(len(x[0]) - 1)
    zs.append(edge)
    xs.append(len(x) - 1)
    ys.append(0)
    zs.append(edge)
    xs.append(len(x) - 1)
    ys.append(len(x[0]) - 1)
    zs.append(edge)
    return xs, ys, zs


def w_g_o(o, r, my_id, eps):
    if isinstance(o, Ship):
        _type = 'ship'
    else:
        _type = 'planet'
    if hasattr(o, 'owner') and getattr(o, 'owner') != None:
        if getattr(o, 'owner') == my_id:
            nef = 'f'
        else:
            nef = 'e'
    else:
        nef = 'u'
    w = weight(o, _type, nef)
    g = gaussian(eps, r)
    return WeightedObject(o.x, o.y, g, w)


def make_axes(length):
    return np.arange(length), np.arange(length), np.arange(length)


def make_grid(x, y):
    return np.meshgrid(x, y, indexing='ij')


def make_data(items, cls='ship', nef='f', color='gray', objs=[]):
    x, y, z = np.array([]), np.array([]), np.array([])
    for t in items:
        x = np.append(x, t.x)
        y = np.append(y, t.y)
        if cls is 'planet':
            r = get_radius(t, 4)
            _z = 0
            spots = t.docking_spots()
            if nef is 'f' and spots:
                _z = -spots
            elif nef is 'e' and spots:
                _z = -len(t.all_docked_ships())
            elif nef is 'u' and spots:
                _z = -spots
            z = np.append(z, t.radius)
            for l in r:
                x = np.append(x, l[0])
                y = np.append(y, l[1])
                z = np.append(z, _z)
        else:
            _z = weight(t, cls, nef)
            z = np.append(z, _z)
            r = get_radius(t, 1)
            for l in r:
                x = np.append(x, l[0])
                y = np.append(y, l[1])
                z = np.append(z, _z)
        objs.append([int(t.x), int(t.y), color, t, cls])
    return x, y, z, objs


def make_rbf(x, y, z, function, smooth, epsilon):
    return Rbf(x, y, z, function=function, smooth=smooth, epsilon=epsilon)


def make_z(F, x, y, skp=1):
    return F(x, y)


def make_grad(Z):
    return np.gradient(Z, axis=(0, 1))


def plotter(Z, ship, objects, turn, pid, width, height):
    af = plt.figure()
    size = af.get_size_inches()
    af.set_size_inches((size[0] * 2, size[1] * 2))
    ax2 = af.add_subplot(211)
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U, V = np.gradient(Z, axis=(0, 1))
    sx, sy = ship[0], ship[1]
    # xx = X[sx, sy]
    # yy = Y[sx, sy]
    # uu = -U[sx, sy]
    # vv = -V[sx, sy]
    # zz = Z[sx, sy]
    ships = [s for s in objects if s[-1] == 'ship']
    # planets = [s for s in o if s[-1] == 'planet']
    items = ships  # + planets
    Xpts, Ypts, Upts, Vpts, Zpts = np.zeros(len(items)), np.zeros(len(items)), np.zeros(len(items)), np.zeros(
        len(items)), np.zeros(len(items))
    for i, l in enumerate(items):
        Xpts[i] = l[0]
        Ypts[i] = l[1]
        Upts[i] = -U[l[0], l[1]]
        Vpts[i] = -V[l[0], l[1]]
        Zpts[i] = Z[l[0], l[1]]
    # skip = (slice(None, None, 3), slice(None, None, 3))
    ax3 = af.add_subplot(212, projection='3d')
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_zlim(-5, 5)
    ax3.scatter(Xpts, Ypts, Zpts, color='purple', alpha=1)
    ax3.plot_surface(X, Y, Z, cmap='terrain')
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.contour(X, Y, Z, cmap='gist_heat')
    # ax2.streamplot(X.T,Y.T,-U.T,-V.T,cmap="magma")
    # ax2.quiver(Xpts,Ypts,Upts,Vpts, cmap='gist_heat', pivot='t	ail',alpha=.5)
    # ax2.quiver(X[skip],Y[skip],-U[skip],-V[skip],Z[skip],alpha=.5,cmap='gist_heat', pivot='tail',angles='xy',scale_units='xy')
    for l in objects:
        if sx == l[0] and sy == l[1]:
            art = Circle((sx, sy), l[3].radius, color='blue')
            ax2.add_artist(art)
            ax2.text(l[0], l[1], l[3].id, color='blue')
        else:
            if l[-1] == 'planet':
                art = Circle((l[0], l[1]), l[3].radius + 2, color='yellow', alpha=.5)
                ax2.add_artist(art)
                art = Circle((l[0], l[1]), l[3].radius, color=l[2], alpha=.5)
                ax2.add_artist(art)
                ax2.text(l[0], l[1], l[3].docking_spots())
            else:
                art = Circle((l[0], l[1]), l[3].radius, color=l[2], alpha=.5)
                ax2.add_artist(art)
                ax2.text(l[0], l[1], l[3].id, color=l[2] if l[2] is 'red' else 'blue')

    plt.savefig("grads/last-{}.png".format(pid))
    plt.clf()
    plt.close()
