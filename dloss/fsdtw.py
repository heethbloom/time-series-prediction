from collections import defaultdict
import numbers

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from torch.autograd import Function

def fsdtw(x, y, radius=1, gamma=0.1, dist=None):
    x, y, dist = __prep_inputs(x, y, dist)
    return __fsdtw(x, y, radius, gamma, dist)


def __difference(a, b):
    return abs(a - b)


def norm(p):
    return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else:
            dist = norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = norm(p=dist)

    return x, y, dist

# 재귀로 구현 (downsampling 반복)
def __fsdtw(x, y, radius, gamma, dist):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, gamma=gamma, window=None, dist=dist)

    x_shrinked = reduce_by_half(x)
    y_shrinked = reduce_by_half(y)
    _, _, path, _, _ = \
        __fsdtw(x_shrinked, y_shrinked, radius=radius, gamma=gamma, dist=dist)
    window = expand_window(path, len(x), len(y), radius)
    return dtw(x, y, gamma=gamma, window=window, dist=dist)



def softmin3(a, b, c, gamma):
    ta = - a / gamma
    tb = - b / gamma
    tc = - c / gamma


    max_val = max(max(ta, tb), tc)

    tmp = 0
    tmp += np.exp(ta - max_val)
    tmp += np.exp(tb - max_val)
    tmp += np.exp(tc - max_val)

    return -gamma * (np.log(tmp) + max_val)


def dtw(x, y, gamma, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    window = list(window)
    D = defaultdict(lambda: (float('inf'), 0, 0))
    Dsoft = defaultdict(lambda: float('inf'))
    D[0, 0] = (0, 0, 0)
    Dsoft[0, 0] = 0

    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        # D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
        #               (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
        D[i, j] = min((Dsoft[i-1, j], i - 1, j), (Dsoft[i, j-1], i, j - 1),
                      (Dsoft[i-1, j-1], i-1, j-1), key=lambda a: a[0])
        t = dt + softmin3(Dsoft[i-1, j], Dsoft[i, j-1], Dsoft[i-1, j-1], gamma)
        Dsoft[i, j] = t
        D[i, j] = (t, D[i, j][1], D[i, j][2])
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return D[len_x, len_y][0], Dsoft[len_x, len_y], path, window, Dsoft


def reduce_by_half(x):
    return np.array([(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)])


###  여기까지 완료 : dtw --> sdtw 가 정확 

def reduce_by_half_filtering(x):
    alpha = 0.01
    tx = np.array(x).squeeze()
    r = list(range(tx.shape[0]-1))
    r[0] = tx[0]
    for i in range(tx.shape[0]-2):
        r[i+1] = (1-alpha) * tx[i] + alpha * r[i]
    return np.array([(r[i] + r[1+i]) / 2 for i in range(0, len(r) - len(r) % 2, 2)])



def expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window


class FSDTW(Function):
    def __init__(self, dist_func=lambda x, y:(x-y)**2, dist_grad_func=None, gamma=0.1, radius=1):
        self.dist_func = dist_func
        self.dist_grad_func = dist_grad_func
        self.gamma = gamma
        self.radius = radius
        self.dtw = []
        self.sdtw = []
        self.windows = []
        self.path = []
        self.R = []
        self.E = []
        self.x = None
        self.y = None
        self.shrinked_x = []
        self.shrinked_y = []
        self.g = []

    def forward(self, x, y):
        x = np.asanyarray(x, dtype='float')
        y = np.asanyarray(y, dtype='float')
        self.x = x
        self.y = y
        dtw, sdtw, _, _, _ = self.__fsdtw(x, y, self.radius, self.gamma, self.dist_func)
        sdtw_weighted = 0
        for i in range(len(self.sdtw)):
            sdtw_weighted += self.sdtw[i] / len(self.shrinked_x[i])
        return self.sdtw[-1] # dtw # sdtw_weighted * len(self.shrinked_x[i]) / len(self.sdtw)

    def __dist(self, i, j):
        try:
            return self.dist_func(self.x[i], self.y[j])
        except IndexError:
            return 0

    def backward(self, grad=0):
        G = None
        g = None
        for i in range(0, len(self.windows)):
            self._grad(self.shrinked_x[i], self.shrinked_y[i], self.R[i], self.windows[i], self.gamma)
            # plt.imshow(self.denseE(i))
            # plt.colorbar()
            # plt.title(f"id {i}")
            # plt.show()
            # print(f"id {i}")
        for i in range(0, len(self.windows)):
            g = self.__jacobian_product_sq_euc(self.shrinked_x[i],
                                               self.shrinked_y[i],
                                               self.E[i],
                                               self.windows[i])
            self.g.append(g)
            # plt.plot(g)
            # plt.title(f"id {i}")
            # plt.show()
            # G = self.__expand_g(G, g).squeeze()
        f = len(self.windows)
        # g = self.__cum_grad()
        g = self.g[-1]
        self._clear_all_state()
        return  g.reshape(-1, 1)

    def __cum_grad(self):
        final_g = self.g[-1].squeeze()
        n = final_g.shape[0]
        for i in range(len(self.g)-1):
            g = self.g[i].squeeze()
            x = np.arange(g.shape[0])
            f = interp1d(x, g)
            xnew = np.linspace(x.min(), x.max(), n)
            ig = f(xnew)
            final_g += ig
        return final_g

    def _expand_g(self, G, g):
        print("注意这里使用的是插值版本")
        if G is None:
            return np.array([g])
        x = np.arange(G.shape[0])
        f = interp1d(x, G)
        xnew = np.linspace(x.min(), x.max(), g.shape[0])
        ig = f(xnew)
        return g.squeeze() + ig


    def denseR(self, i):
        m, n = self.windows[i][-1]
        R = np.zeros((m + 2, n + 2))
        for (i, j), v in self.R[i].items():
            R[i, j] = v
        return R

    def denseE(self, i=-1):
        m, n = self.windows[i][-1]
        E = np.zeros((m+2, n+2))
        for (i, j), v in self.E[i].items():
            E[i, j] = v
        return E


    def _grad(self, x, y, R_, window, gamma):
        def dist(i, j):
            try:
                return self.dist_func(x[i], y[j])
            except IndexError:
                return 0
        m, n = window[-1]

        E = defaultdict(lambda: 0)
        R = defaultdict(lambda: float('-inf'))
        for (i, j), v in R_.items():
            if v != float('inf'):
                R[i, j] = v
            # R[i, j] = v
        for i in range(1, m + 1):
            R[i, n + 1] = -float('inf')

        for j in range(1, n + 1):
            R[m + 1, j] = -float('inf')
        E[m+1, n+1] = 1
        R[m + 1, n + 1] = R[m, n]
        for i, j in reversed(window):
            # print(i, j)
            if i < 1 or j < 1:
                continue
            a = np.exp((R[i + 1, j] - R[i, j] - dist(i, j - 1)) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - dist(i - 1, j)) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - dist(i, j)) / gamma)
            t = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
            E[i, j] = t
        self.E.append(E)

    def __jacobian_product_sq_euc(self, x, y, E, window):
        G_ = defaultdict(lambda: 0)
        for i, j in window:
            i -= 1
            j -= 1
            t = E[i+1, j+1] * 2 * (x[i] - y[j])
            G_[i] = G_[i] + t
        G = np.zeros(x.shape)
        for k, v in G_.items():
            G[k] = v
        return G

    def _save_state(self, sx, sy, distance, soft_distance, path, window, Dsoft):
        self.shrinked_x.append(sx)
        self.shrinked_y.append(sy)
        self.dtw.append(distance)
        self.sdtw.append(soft_distance)
        self.path.append(path)
        self.windows.append(list(window))
        self.R.append(Dsoft)

    def _clear_all_state(self):
        self.shrinked_x = []
        self.shrinked_y = []
        self.dtw = []
        self.sdtw = []
        self.path = []
        self.windows = []
        self.R = []
        self.g = []

    def __fsdtw(self, x, y, radius, gamma, dist):
        min_time_size = radius + 2

        if len(x) < min_time_size or len(y) < min_time_size:
            state = dtw(x, y, gamma=gamma, window=None, dist=dist)
            self._save_state(x, y, *state)
            return state

        x_shrinked = reduce_by_half(x)
        y_shrinked = reduce_by_half(y)
        distance, soft_distance, path, _, Dsoft = \
            self.__fsdtw(x_shrinked, y_shrinked, radius=radius, gamma=gamma, dist=dist)
        hr_window = expand_window(path, len(x), len(y), radius)
        state = dtw(x, y, gamma=gamma, window=hr_window, dist=dist)
        self._save_state(x, y, *state)
        return state


if __name__ == "__main__":
    fsdtw = FSDTW(norm(2), None, gamma=0.1)
    x = np.array([1, 2, 3, 4, 5], dtype='float')
    y = np.array([2, 3, 4], dtype='float')
    sdtw = fsdtw.forward(x, y)
    fsdtw.backward(0)
    print(sdtw)
    # print(fsdtw(x, y, dist=2))
    # (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])