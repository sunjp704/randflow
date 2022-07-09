from numpy import vstack


class Spectrum(object):

    @staticmethod
    def davenport():
        pass

    @staticmethod
    def simu():
        pass

    @staticmethod
    def hino():
        pass

    @staticmethod
    def harris():
        pass

    @staticmethod
    def vonKarman(u_avg, turbulence, fn):
        Iu, Iv, Iw, Lu, Lv, Lw = turbulence
        Su = 4 * (Iu * u_avg)**2 * Lu / u_avg / (1 + 70.8 *
                                                 (fn * Lu / u_avg)**2)**(5 / 6)
        Sv = 4 * (Iv * u_avg)**2 * Lv / u_avg * (
            1 + 188.4 *
            (2 * fn * Lv / u_avg)**2) / (1 + 70.8 *
                                         (fn * Lv / u_avg)**2)**(11 / 6)
        Sw = 4 * (Iw * u_avg)**2 * Lw / u_avg * (
            1 + 188.4 *
            (2 * fn * Lw / u_avg)**2) / (1 + 70.8 *
                                         (fn * Lw / u_avg)**2)**(11 / 6)

        return vstack((Su, Sv, Sw))
