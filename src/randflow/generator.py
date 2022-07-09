from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .point import Point


class Generator(ABC):

    @abstractmethod
    def create_fluct():
        pass


class DSRFG(Generator):
    pass


class CDRFG(Generator):
    pass


class NSRFG(Generator):

    def __init__(self,
                 *,
                 N: int,
                 fmax: Union[int, float],
                 target_spectrum: str,
                 c: list = [8, 10, 15],
                 gamma: list = [3.2, 1.6, 1.4]) -> None:
        self.N = N
        self.fmax = fmax
        self.delta_f = fmax / N
        self.fn = np.linspace(0.5 * self.delta_f,
                              (2 * N - 1) / 2 * self.delta_f, N)
        self.target_spectrum = target_spectrum
        self.c = c
        self.gamma = gamma

    def create_fluct(self, time, point: Point, seed=(None, None)):
        target_spec = point.create_spectrum(self.target_spectrum, self.fn)
        c = np.array(self.c).reshape((3, 1))
        gamma = np.array(self.gamma).reshape((3, 1))
        L = point.u_avg / (self.fn * (c * gamma))
        x_wave = np.array([point.x, point.y, point.z]).reshape((3, 1)) / L
        p = np.sqrt(2 * target_spec * self.delta_f)
        q = p / L
        A = np.sqrt((q[1, :]**2 + q[2, :]**2)**2 + q[0, :]**2 * q[1, :]**2 +
                    q[0, :]**2 * q[2, :]**2)
        B = np.sqrt(q[1, :]**2 + q[2, :]**2)
        rng = np.random.default_rng(seed[0])
        theta = 2 * np.pi * rng.random((self.N, ))
        k1 = -(q[1, :]**2 + q[2, :]**2) / A * np.sin(theta)
        k2 = q[0, :] * q[1, :] / A * np.sin(theta) + q[2, :] / B * np.cos(
            theta)
        k3 = q[0, :] * q[2, :] / A * np.sin(theta) - q[1, :] / B * np.cos(
            theta)
        k = np.vstack((k1, k2, k3))
        rng = np.random.default_rng(seed[1])
        phi = 2 * np.pi * rng.random((self.N, ))
        fluct = np.empty((3, len(time)), dtype=float)
        for count in range(len(time)):
            fluct[:, count] = np.sum(
                p *
                np.sin(k * x_wave + 2 * np.pi * self.fn * time[count] + phi),
                axis=1)
        return (fluct, target_spec)


class GeneratorFactory(object):

    @staticmethod
    def create_generator(generator: str, kwargs: dict):
        """

        Args:
            generator: 'DSRFG' or 'DSRFG' or 'CDRFG'
            kwargs: dict of arguments passed to initialize a generator
                    for NSRFG, ={N: int,
                                 fmax: Union[int, float],
                                 target_spectrum: str,
                                 c: list = [8, 10, 15],
                                 gamma: list = [3.2, 1.6, 1.4]}
                    for DSRFG, ={}
                    for CDRFG, ={}


        """
        try:

            if generator == 'DSRFG':
                return DSRFG(**kwargs)
            elif generator == 'CDRFG':
                return CDRFG(**kwargs)
            elif generator == 'NSRFG':
                return NSRFG(**kwargs)
            else:
                raise RuntimeError('Generator' + generator + 'not defined')
        except RuntimeError as e:
            print(repr(e))
