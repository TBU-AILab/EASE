"""
CEC 2017 single objective bound constrained benchmark (also used in CEC 2024 and CEC 2026 competitions).

Python port of the official C code cec17_test_func.cpp (N. Awad et al.,
https://github.com/P-N-Suganthan/CEC2017-BoundContrained). The port intentionally mirrors the C implementation,
including its side effects on the shared work buffers y and z (e.g. Schaffer F7 reads y, hybrid functions pass
sub-vectors of y to their components), so that function values match the official code. Verified against the
compiled official code for all functions at D=30.

Only the input data for D=30 are bundled (input_data/), other dimensions require the corresponding data files.
"""

import math
import os

import numpy as np

INF = 1.0e99
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029

CF_NUM = 10  # number of shift vectors / matrices stored for composition functions
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data")


class CEC2017Engine:
    """
    Evaluation of one CEC 2017 function (func_num) in dimension nx with the official data.
    """

    def __init__(self, func_num: int, nx: int, data_dir: str = DATA_DIR):
        if not 1 <= func_num <= 30:
            raise ValueError("There are only 30 test functions in this test suite.")
        self.func_num = func_num
        self.nx = nx

        # work buffers shared by all functions (global y, z in the C code)
        self.y = np.zeros(nx)
        self.z = np.zeros(nx)

        m = np.loadtxt(os.path.join(data_dir, f"M_{func_num}_D{nx}.txt")).ravel()
        if func_num < 20:
            self.M = m[: nx * nx].reshape(nx, nx)
        else:
            self.M = m[: CF_NUM * nx * nx].reshape(-1, nx, nx)

        shift = np.atleast_2d(
            np.loadtxt(os.path.join(data_dir, f"shift_data_{func_num}.txt"))
        )
        if func_num < 20:
            self.OShift = shift.ravel()[:nx].copy()
        else:
            self.OShift = shift[:CF_NUM, :nx].ravel().copy()

        self.SS = None
        if 11 <= func_num <= 20:
            self.SS = np.loadtxt(
                os.path.join(data_dir, f"shuffle_data_{func_num}_D{nx}.txt"), dtype=int
            ).ravel()[:nx]
        elif func_num in (29, 30):
            self.SS = np.loadtxt(
                os.path.join(data_dir, f"shuffle_data_{func_num}_D{nx}.txt"), dtype=int
            ).ravel()[: nx * CF_NUM]

    ####################################################################
    #########  Evaluation
    ####################################################################
    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        nx, Os, M, SS = self.nx, self.OShift, self.M, self.SS
        k = self.func_num
        base = {
            1: self.bent_cigar,
            2: self.sum_diff_pow,
            3: self.zakharov,
            4: self.rosenbrock,
            5: self.rastrigin,
            6: self.schaffer_F7,
            7: self.bi_rastrigin,
            8: self.step_rastrigin,
            9: self.levy,
            10: self.schwefel,
        }
        hybrids = {
            11: self.hf01,
            12: self.hf02,
            13: self.hf03,
            14: self.hf04,
            15: self.hf05,
            16: self.hf06,
            17: self.hf07,
            18: self.hf08,
            19: self.hf09,
            20: self.hf10,
        }
        if k in base:
            f = base[k](x, nx, Os, M, 1, 1)
        elif k in hybrids:
            # F20 loads CF_NUM matrices (func_num >= 20) but uses only the first one
            f = hybrids[k](x, nx, Os, M if M.ndim == 2 else M[0], SS, 1, 1)
        elif k in (29, 30):
            f = (self.cf09 if k == 29 else self.cf10)(x, nx, Os, M, SS, 1)
        else:
            f = getattr(self, f"cf{k - 20:02d}")(x, nx, Os, M, 1)
        return f + 100.0 * k

    ####################################################################
    #########  Transformations
    ####################################################################
    def sr_func(self, x, sr_x, nx, Os, Mr, sh_rate, s_flag, r_flag):
        """Shift and rotate. Writes the result into sr_x (and y when rotating), as the C code."""
        y = self.y
        if s_flag == 1:
            if r_flag == 1:
                y[:nx] = (x[:nx] - Os[:nx]) * sh_rate
                sr_x[:nx] = Mr @ y[:nx]
            else:
                sr_x[:nx] = (x[:nx] - Os[:nx]) * sh_rate
        else:
            if r_flag == 1:
                y[:nx] = x[:nx] * sh_rate
                sr_x[:nx] = Mr @ y[:nx]
            else:
                sr_x[:nx] = x[:nx] * sh_rate

    ####################################################################
    #########  Basic functions
    ####################################################################
    def ellips(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        i = np.arange(nx)
        return float(np.sum(np.power(10.0, 6.0 * i / (nx - 1)) * z[:nx] * z[:nx]))

    def sum_diff_pow(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        return float(np.sum(np.power(np.abs(z[:nx]), np.arange(1, nx + 1))))

    def zakharov(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        zz = z[:nx]
        sum1 = np.sum(zz**2)
        sum2 = np.sum(0.5 * np.arange(1, nx + 1) * zz)
        return float(sum1 + sum2**2 + sum2**4)

    def levy(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        w = 1.0 + (z[:nx] - 1.0) / 4.0
        term1 = math.sin(PI * w[0]) ** 2
        term3 = (w[nx - 1] - 1) ** 2 * (1 + math.sin(2 * PI * w[nx - 1]) ** 2)
        wi = w[: nx - 1]
        # sin(PI*wi+1) as in the official code
        s = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(PI * wi + 1) ** 2))
        return float(term1 + s + term3)

    def bent_cigar(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        return float(z[0] * z[0] + np.sum(1e6 * z[1:nx] * z[1:nx]))

    def discus(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        return float(1e6 * z[0] * z[0] + np.sum(z[1:nx] * z[1:nx]))

    def rosenbrock(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag)
        zz = z[:nx] + 1.0
        tmp1 = zz[:-1] * zz[:-1] - zz[1:]
        tmp2 = zz[:-1] - 1.0
        return float(np.sum(100.0 * tmp1 * tmp1 + tmp2 * tmp2))

    def schaffer_F7(self, x, nx, Os, Mr, s_flag, r_flag):
        # NOTE: the official code evaluates the (unrotated) buffer y, not z
        z, y = self.z, self.y
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        zi = np.sqrt(y[: nx - 1] ** 2 + y[1:nx] ** 2)
        z[: nx - 1] = zi
        tmp = np.sin(50.0 * zi**0.2)
        f = np.sum(zi**0.5 + zi**0.5 * tmp * tmp)
        return float(f * f / (nx - 1) / (nx - 1))

    def ackley(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        sum1 = -0.2 * math.sqrt(np.sum(z[:nx] ** 2) / nx)
        sum2 = np.sum(np.cos(2.0 * PI * z[:nx])) / nx
        return float(E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0)

    _W_A = 0.5 ** np.arange(21)
    _W_B = 3.0 ** np.arange(21)

    def weierstrass(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 0.5 / 100.0, s_flag, r_flag)
        a, b = self._W_A, self._W_B
        s = np.sum(a[None, :] * np.cos(2.0 * PI * b[None, :] * (z[:nx, None] + 0.5)))
        sum2 = np.sum(a * np.cos(2.0 * PI * b * 0.5))
        return float(s - nx * sum2)

    def griewank(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag)
        s = np.sum(z[:nx] ** 2)
        p = np.prod(np.cos(z[:nx] / np.sqrt(1.0 + np.arange(nx))))
        return float(1.0 + s / 4000.0 - p)

    def rastrigin(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
        zz = z[:nx]
        return float(np.sum(zz * zz - 10.0 * np.cos(2.0 * PI * zz) + 10.0))

    def step_rastrigin(self, x, nx, Os, Mr, s_flag, r_flag):
        # The official code modifies y before sr_func, which overwrites it again; the result equals Rastrigin
        return self.rastrigin(x, nx, Os, Mr, s_flag, r_flag)

    def schwefel(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag)
        zz = z[:nx] + 4.209687462275036e002
        f = np.zeros(nx)
        hi = zz > 500
        lo = zz < -500
        mid = ~(hi | lo)
        m = 500.0 - np.fmod(zz[hi], 500)
        f[hi] = -m * np.sin(np.power(m, 0.5)) + ((zz[hi] - 500.0) / 100) ** 2 / nx
        a = np.fmod(np.abs(zz[lo]), 500)
        f[lo] = (
            -(-500.0 + a) * np.sin(np.power(500.0 - a, 0.5))
            + ((zz[lo] + 500.0) / 100) ** 2 / nx
        )
        f[mid] = -zz[mid] * np.sin(np.power(np.abs(zz[mid]), 0.5))
        return float(np.sum(f) + 4.189828872724338e002 * nx)

    _K_P = 2.0 ** np.arange(1, 33)

    def katsuura(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        tmp3 = math.pow(1.0 * nx, 1.2)
        self.sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
        tmp2 = self._K_P[None, :] * z[:nx, None]
        temp = np.sum(np.abs(tmp2 - np.floor(tmp2 + 0.5)) / self._K_P[None, :], axis=1)
        f = np.prod(np.power(1.0 + np.arange(1, nx + 1) * temp, 10.0 / tmp3))
        tmp1 = 10.0 / nx / nx
        return float(f * tmp1 - tmp1)

    def bi_rastrigin(self, x, nx, Os, Mr, s_flag, r_flag):
        z, y = self.z, self.y
        mu0, d = 2.5, 1.0
        s = 1.0 - 1.0 / (2.0 * math.pow(nx + 20.0, 0.5) - 8.2)
        mu1 = -math.pow((mu0 * mu0 - d) / s, 0.5)

        if s_flag == 1:
            y[:nx] = x[:nx] - Os[:nx]
        else:
            y[:nx] = np.array(x[:nx])  # x may alias y (hybrid functions)
        y[:nx] *= 10.0 / 100.0

        tmpx = 2 * y[:nx]
        tmpx[Os[:nx] < 0.0] *= -1.0
        z[:nx] = tmpx
        tmpx = tmpx + mu0
        tmp1 = np.sum((tmpx - mu0) ** 2)
        tmp2 = np.sum((tmpx - mu1) ** 2) * s + d * nx

        if r_flag == 1:
            y[:nx] = Mr @ z[:nx]
            tmp = np.sum(np.cos(2.0 * PI * y[:nx]))
        else:
            tmp = np.sum(np.cos(2.0 * PI * z[:nx]))
        f = tmp1 if tmp1 < tmp2 else tmp2
        return float(f + 10.0 * (nx - tmp))

    def grie_rosen(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
        zz = z[:nx] + 1.0
        nxt = np.roll(zz, -1)  # pairs (i, i+1) and the wrap-around pair (nx-1, 0)
        tmp1 = zz * zz - nxt
        tmp2 = zz - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        return float(np.sum((temp * temp) / 4000.0 - np.cos(temp) + 1.0))

    def escaffer6(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        self.sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)
        zz = z[:nx]
        nxt = np.roll(zz, -1)
        r2 = zz * zz + nxt * nxt
        temp1 = np.sin(np.sqrt(r2)) ** 2
        temp2 = 1.0 + 0.001 * r2
        return float(np.sum(0.5 + (temp1 - 0.5) / (temp2 * temp2)))

    def happycat(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        alpha = 1.0 / 8.0
        self.sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
        z[:nx] -= 1.0
        r2 = np.sum(z[:nx] ** 2)
        sum_z = np.sum(z[:nx])
        return float(math.pow(abs(r2 - nx), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5)

    def hgbat(self, x, nx, Os, Mr, s_flag, r_flag):
        z = self.z
        alpha = 1.0 / 4.0
        self.sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
        z[:nx] -= 1.0
        r2 = np.sum(z[:nx] ** 2)
        sum_z = np.sum(z[:nx])
        return float(
            math.pow(abs(r2**2.0 - sum_z**2.0), 2 * alpha)
            + (0.5 * r2 + sum_z) / nx
            + 0.5
        )

    ####################################################################
    #########  Hybrid functions
    ####################################################################
    def _hybrid(self, x, nx, Os, Mr, S, s_flag, r_flag, Gp, components):
        cf_num = len(Gp)
        G_nx = [math.ceil(Gp[i] * nx) for i in range(cf_num - 1)]
        G_nx.append(nx - sum(G_nx))
        G = [0]
        for i in range(1, cf_num):
            G.append(G[i - 1] + G_nx[i - 1])

        self.sr_func(x, self.z, nx, Os, Mr, 1.0, s_flag, r_flag)
        self.y[:nx] = self.z[S[:nx] - 1]

        f = 0.0
        for i, comp in enumerate(components):
            # components get a view into y (pointer &y[G[i]] in the C code)
            f += comp(self.y[G[i] : G[i] + G_nx[i]], G_nx[i], Os, Mr, 0, 0)
        return f

    def hf01(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.4, 0.4],
            [self.zakharov, self.rosenbrock, self.rastrigin],
        )

    def hf02(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.3, 0.3, 0.4],
            [self.ellips, self.schwefel, self.bent_cigar],
        )

    def hf03(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.3, 0.3, 0.4],
            [self.bent_cigar, self.rosenbrock, self.bi_rastrigin],
        )

    def hf04(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.2, 0.2, 0.4],
            [self.ellips, self.ackley, self.schaffer_F7, self.rastrigin],
        )

    def hf05(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.2, 0.3, 0.3],
            [self.bent_cigar, self.hgbat, self.rastrigin, self.rosenbrock],
        )

    def hf06(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.2, 0.3, 0.3],
            [self.escaffer6, self.hgbat, self.rosenbrock, self.schwefel],
        )

    def hf07(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.1, 0.2, 0.2, 0.2, 0.3],
            [
                self.katsuura,
                self.ackley,
                self.grie_rosen,
                self.schwefel,
                self.rastrigin,
            ],
        )

    def hf08(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [self.ellips, self.ackley, self.rastrigin, self.hgbat, self.discus],
        )

    def hf09(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [
                self.bent_cigar,
                self.rastrigin,
                self.grie_rosen,
                self.weierstrass,
                self.escaffer6,
            ],
        )

    def hf10(self, x, nx, Os, Mr, S, s_flag, r_flag):
        return self._hybrid(
            x,
            nx,
            Os,
            Mr,
            S,
            s_flag,
            r_flag,
            [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
            [
                self.hgbat,
                self.katsuura,
                self.ackley,
                self.rastrigin,
                self.schwefel,
                self.schaffer_F7,
            ],
        )

    ####################################################################
    #########  Composition functions
    ####################################################################
    @staticmethod
    def cf_cal(x, nx, Os, delta, bias, fit):
        cf_num = len(fit)
        fit = [fit[i] + bias[i] for i in range(cf_num)]
        w = []
        for i in range(cf_num):
            wi = float(np.sum((x[:nx] - Os[i * nx : (i + 1) * nx]) ** 2.0))
            if wi != 0:
                wi = math.pow(1.0 / wi, 0.5) * math.exp(
                    -wi / 2.0 / nx / math.pow(delta[i], 2.0)
                )
            else:
                wi = INF
            w.append(wi)
        w_max = max(w)
        w_sum = sum(w)
        if w_max == 0:
            w = [1.0] * cf_num
            w_sum = cf_num
        f = 0.0
        for i in range(cf_num):
            f = f + w[i] / w_sum * fit[i]
        return f

    def _composition(self, x, nx, Os, Mr, r_flag, delta, bias, components):
        fit = []
        for i, (comp, scale) in enumerate(components):
            fi = comp(x, nx, Os[i * nx : (i + 1) * nx], Mr[i], 1, r_flag)
            if scale is not None:
                fi = scale(fi)
            fit.append(fi)
        return self.cf_cal(x, nx, Os, delta, bias, fit)

    def cf01(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30],
            [0, 100, 200],
            [
                (self.rosenbrock, None),
                (self.ellips, lambda f: 10000 * f / 1e10),
                (self.rastrigin, None),
            ],
        )

    def cf02(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30],
            [0, 100, 200],
            [
                (self.rastrigin, None),
                (self.griewank, lambda f: 1000 * f / 100),
                (self.schwefel, None),
            ],
        )

    def cf03(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30, 40],
            [0, 100, 200, 300],
            [
                (self.rosenbrock, None),
                (self.ackley, lambda f: 1000 * f / 100),
                (self.schwefel, None),
                (self.rastrigin, None),
            ],
        )

    def cf04(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30, 40],
            [0, 100, 200, 300],
            [
                (self.ackley, lambda f: 1000 * f / 100),
                (self.ellips, lambda f: 10000 * f / 1e10),
                (self.griewank, lambda f: 1000 * f / 100),
                (self.rastrigin, None),
            ],
        )

    def cf05(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30, 40, 50],
            [0, 100, 200, 300, 400],
            [
                (self.rastrigin, lambda f: 10000 * f / 1e3),
                (self.happycat, lambda f: 1000 * f / 1e3),
                (self.ackley, lambda f: 1000 * f / 100),
                (self.discus, lambda f: 10000 * f / 1e10),
                (self.rosenbrock, None),
            ],
        )

    def cf06(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 20, 30, 40],
            [0, 100, 200, 300, 400],
            [
                (self.escaffer6, lambda f: 10000 * f / 2e7),
                (self.schwefel, None),
                (self.griewank, lambda f: 1000 * f / 100),
                (self.rosenbrock, None),
                (self.rastrigin, lambda f: 10000 * f / 1e3),
            ],
        )

    def cf07(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30, 40, 50, 60],
            [0, 100, 200, 300, 400, 500],
            [
                (self.hgbat, lambda f: 10000 * f / 1000),
                (self.rastrigin, lambda f: 10000 * f / 1e3),
                (self.schwefel, lambda f: 10000 * f / 4e3),
                (self.bent_cigar, lambda f: 10000 * f / 1e30),
                (self.ellips, lambda f: 10000 * f / 1e10),
                (self.escaffer6, lambda f: 10000 * f / 2e7),
            ],
        )

    def cf08(self, x, nx, Os, Mr, r_flag):
        return self._composition(
            x,
            nx,
            Os,
            Mr,
            r_flag,
            [10, 20, 30, 40, 50, 60],
            [0, 100, 200, 300, 400, 500],
            [
                (self.ackley, lambda f: 1000 * f / 100),
                (self.griewank, lambda f: 1000 * f / 100),
                (self.discus, lambda f: 10000 * f / 1e10),
                (self.rosenbrock, None),
                (self.happycat, lambda f: 1000 * f / 1e3),
                (self.escaffer6, lambda f: 10000 * f / 2e7),
            ],
        )

    def _composition_hybrid(self, x, nx, Os, Mr, SS, r_flag, hybrids):
        fit = [
            hf(
                x,
                nx,
                Os[i * nx : (i + 1) * nx],
                Mr[i],
                SS[i * nx : (i + 1) * nx],
                1,
                r_flag,
            )
            for i, hf in enumerate(hybrids)
        ]
        return self.cf_cal(x, nx, Os, [10, 30, 50], [0, 100, 200], fit)

    def cf09(self, x, nx, Os, Mr, SS, r_flag):
        return self._composition_hybrid(
            x, nx, Os, Mr, SS, r_flag, [self.hf05, self.hf06, self.hf07]
        )

    def cf10(self, x, nx, Os, Mr, SS, r_flag):
        return self._composition_hybrid(
            x, nx, Os, Mr, SS, r_flag, [self.hf05, self.hf08, self.hf09]
        )


class CEC2017function:
    """
    CEC 2017 function for the metaheuristic evaluators.
    evaluate() returns the error f(x) - f(x*), where f(x*) = 100 * funcNum.
    """

    def __init__(self, funcNum: int, dim: int = 30):
        self.func = funcNum
        self.dim = dim
        self._engine = CEC2017Engine(funcNum, dim)
        self.optimum_value = 100.0 * funcNum

    def get_bounds(self):
        return np.array([[-100.0, 100.0]] * self.dim)

    def evaluate(self, x) -> float:
        return self._engine.evaluate(x) - self.optimum_value

    def __str__(self) -> str:
        return f"CEC2017-F{self.func}-D{self.dim}"
