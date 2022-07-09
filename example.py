import concurrent.futures

import numpy as np
from scipy.signal import welch
from matplotlib import pyplot as plt

from src.randflow import GeneratorFactory, Point


def main():
    x = np.arange(5) / 10
    plist = [
        Point('p' + str(i), x[i], 0, 2, 10, (0.08, 0.04, 0.04, 0.6, 0.3, 0.3))
        for i in range(5)
    ]
    fs = 200
    time = np.arange(0, 10 + 1 / fs, 1 / fs)
    gen = GeneratorFactory.create_generator('NSRFG', {
        'N': 1000,
        'fmax': 100,
        'target_spectrum': 'vonKarman'
    })
    fluct = dict()
    spec = dict()
    # for p in plist:
    #     fluct[p], spec[p] = gen.create_fluct(time, p)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for p in plist:
            out = executor.submit(gen.create_fluct, time, p)
            fluct[p], spec[p] = out.result()

    plot_point = plist[0]
    [u, v, w] = [fluct[plot_point][i] for i in range(3)]
    [Su, Sv, Sw] = [spec[plot_point][i] for i in range(3)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5))
    ax1.plot(time, u, 'g-')
    ax1.set_xlabel('t(s)')
    ax1.set_ylabel(r'$u(m\cdot s^{-1})$')
    ax2.plot(time, v, 'g-')
    ax2.set_xlabel('t(s)')
    ax2.set_ylabel(r'$v(m\cdot s^{-1})$')
    ax3.plot(time, w, 'g-')
    ax3.set_xlabel('t(s)')
    ax3.set_ylabel(r'$w(m\cdot s^{-1})$')
    fig.suptitle('velocity time history of ' + plot_point.name)

    length = len(u)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    f, pxx = welch(u, fs, window='hamming', nperseg=length // 4)
    ax1.loglog(f, pxx, label='NSRFG')
    ax1.loglog(gen.fn, Su, label=gen.target_spectrum)
    ax1.set_xlabel('Frequency(Hz)')
    ax1.set_ylabel(r'$S_u(m^2s^2\cdot Hz^{-1})$')
    ax1.legend()
    f, pxx = welch(v, fs, window='hamming', nperseg=length // 4)
    ax2.loglog(f, pxx, label='NSRFG')
    ax2.loglog(gen.fn, Sv, label=gen.target_spectrum)
    ax2.set_xlabel('Frequency(Hz)')
    ax2.set_ylabel(r'$S_v(m^2s^2\cdot Hz^{-1})$')
    ax2.legend()
    f, pxx = welch(w, fs, window='hamming', nperseg=length // 4)
    ax3.loglog(f, pxx, label='NSRFG')
    ax3.loglog(gen.fn, Sw, label=gen.target_spectrum)
    ax3.set_xlabel('Frequency(Hz)')
    ax3.set_ylabel(r'$S_w(m^2s^2\cdot Hz^{-1})$')
    ax3.legend()

    plt.show()


if __name__ == '__main__':
    main()
