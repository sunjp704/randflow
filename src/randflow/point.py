from typing import Tuple

from .spectrum import Spectrum


class Point(object):
    def __init__(self, name, loction: Tuple[float, float, float], u_avg, turbulence: Tuple[float, float, float, float, float,
                                                                                           float]) -> None:
        self.name = name
        self.x = loction[0]
        self.y = loction[1]
        self.z = loction[2]
        self.u_avg = u_avg
        self.turb = turbulence

    def create_spectrum(self, target_spectrum: str, fn):
        self.target_spectrum = target_spectrum
        if target_spectrum == 'Davenport':
            return Spectrum.davenport()
        elif target_spectrum == 'Kaimal':
            return Spectrum.kaimal()
        elif target_spectrum == 'vonKarman':
            return Spectrum.vonKarman(self.u_avg, self.turb, fn)
