import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa, librosa.display
from .features import *

plt.rcParams['figure.figsize'] = (10, 3)


def show_audio(path_audio, sr=22050, duration=30, start=10, trim=True, figsize=(10, 3), listen=True):
    """
    show the audio info, plot, listen, and return y
    :param path_audio: audio path
    :param sr: sampling rate
    :param duration: duration time (in seconds)
    :param start: start time (in seconds)
    :param trim: True or False
    :param figsize: (10, 3)
    :param listen:
    :return: y
    """
    y = get_y_from_audio(path_audio=path_audio, sr=sr, duration=duration, start=start, trim=trim)
    print(f"- Digital signals information - \n audio time series length: {len(y)} \n sampling rate: {sr} \n audio length: {len(y) / sr} seconds")

    f, ax = plt.subplots(figsize=figsize)
    ax.set(title='Waveform')
    librosa.display.waveshow(y=y, sr=sr, ax=ax)
    plt.show()
    if listen is True:
        ipd.display(ipd.Audio(y, rate=sr))
    return y


def show_hpss(y, y_harm, y_perc, sr=22050, title='Harmonic-Percussive Separation', figsize=(10, 3), listen=True):
    f, ax = plt.subplots(figsize=figsize)
    ax.set(title=title)
    librosa.display.waveshow(y=y, sr=sr, alpha=.5, label='original',ax=ax)
    librosa.display.waveshow(y=y_harm, sr=sr, label='harmonic',ax=ax)
    librosa.display.waveshow(y=y_perc, sr=sr, label='percussive',ax=ax)
    ax.legend()
    plt.show()

    if listen is True:
        print("Harmonic")
        ipd.display(ipd.Audio(data=y_harm, rate=sr))
        print("Percussive")
        ipd.display(ipd.Audio(data=y_perc, rate=sr))


def show_spectrogram(S, sr=22050, title='log Spectrogram', x_axis='time', y_axis='log', figsize=(10, 3)):
    """
    :param S:
    :param sr:
    :param title:
    :param x_axis:
    :param y_axis:
    :param figsize:
    :return:
    """
    f, ax = plt.subplots(figsize=figsize)
    ax.set(title=title)
    img = librosa.display.specshow(data=S, sr=sr, x_axis=x_axis, y_axis=y_axis, ax=ax)
    f.colorbar(img, ax=ax)
    plt.show()


def show_spectral_centroids(y, sr=22050, title='log Spectrogram & spectral centroids', figsize=(10, 3)):
    spec_centr = get_spectral_centroids(y=y, sr=sr)
    S_dB = stft_mag(y=y, power='energy', return_dB=True)
    times = librosa.times_like(spec_centr)
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.specshow(S_dB, y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, spec_centr.T, label='Spectral Centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title=title)
    plt.show()


def show_spectral_bandwidth(y, sr=22050, figsize=(10, 6)):
    spec_bw = get_spectral_bandwidth(y=y, sr=sr)
    times = librosa.times_like(spec_bw)
    spec_centr = get_spectral_centroids(y=y, sr=sr)
    S_dB = stft_mag(y, power='energy', return_dB=True)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    ax[0].plot(times, spec_bw, label='Spectral bandwidth')
    ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(S_dB, y_axis='log', x_axis='time', ax=ax[1])
    ax[1].fill_between(times, np.maximum(0, spec_centr - spec_bw),
                       np.minimum(spec_centr + spec_bw, sr / 2),
                       alpha=0.6, label='Centroid +- bandwidth', color='y')
    ax[1].plot(times, spec_centr, label='Spectral centroid', color='w')
    ax[1].legend(loc='lower right')
    ax[1].set(title='log Spectrogram')
    plt.show()


def show_spectral_contrast(y, n_bands=4, figsize=(10, 3)):
    spec_contrast = get_spectral_contrast(y=y, n_bands=n_bands)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(spec_contrast, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(ylabel='Frequency bands', title='Spectral Contrast')
    plt.show()


def show_spectral_flatness(y, title='Spectral Flatness & log Spectrogram', figsize=(10, 6)):
    spec_flat = get_spectral_flatness(y=y)
    times = librosa.times_like(spec_flat)
    S_dB = stft_mag(y, power='energy', return_dB=True)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    plt.suptitle(title)
    ax[0].plot(times, spec_flat, label='Spectral Flatness')
    ax[0].set(ylabel='Flatness')
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(S_dB, y_axis='log', x_axis='time', ax=ax[1])
    plt.show()


def show_spectral_rolloff(y, roll_percent=0.85, title='Spectral Roll-offs & log Spectrogram', figsize=(10, 3)):
    spec_rolloff = get_spectral_rolloff(y, roll_percent=roll_percent)
    spec_rolloff_max = get_spectral_rolloff(y, roll_percent=0.99)
    spec_rolloff_min = get_spectral_rolloff(y, roll_percent=0.01)
    times = librosa.times_like(spec_rolloff)
    S_dB = stft_mag(y, power='energy', return_dB=True)

    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.specshow(S_dB,
                              y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, spec_rolloff_max, label='Roll-off frequency (0.99)')
    ax.plot(times, spec_rolloff, label=f'Roll-off frequency ({roll_percent})')
    ax.plot(times, spec_rolloff_min, color='w', label='Roll-off frequency (0.01)')
    ax.legend(loc='lower right')
    ax.set(title=title)
    plt.show()


def show_spectral_poly(y, title='Polynomial Spectral Coefficients', figsize=(10, 8)):
    # Fit a degree-0 polynomial (constant) to each frame
    spec_poly_0 = get_spectral_poly(y, order=0)
    # Fit a linear polynomial to each frame
    spec_poly_1 = get_spectral_poly(y, order=1)
    # Fit a quadratic to each frame
    spec_poly_2 = get_spectral_poly(y, order=2)
    times = librosa.times_like(spec_poly_0)
    S_dB = stft_mag(y, power='energy', return_dB=True)

    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=figsize)
    plt.suptitle(title)
    ax[0].plot(times, spec_poly_0[0], label='order=0', alpha=0.8, c='m')
    ax[0].plot(times, spec_poly_1[1], label='order=1', alpha=0.8, c='y')
    ax[0].plot(times, spec_poly_2[2], label='order=2', alpha=0.8, c='c')
    ax[0].legend()
    ax[0].label_outer()
    ax[0].set(ylabel='Constant term ')
    ax[1].plot(times, spec_poly_1[0], label='order=1', alpha=0.8, c='y')
    ax[1].plot(times, spec_poly_2[1], label='order=2', alpha=0.8, c='c')
    ax[1].set(ylabel='Linear term')
    ax[1].label_outer()
    ax[1].legend()
    ax[2].plot(times, spec_poly_2[0], label='order=2', alpha=0.8, c='c')
    ax[2].set(ylabel='Quadratic term')
    ax[2].legend()
    librosa.display.specshow(S_dB,
                             y_axis='log', x_axis='time', ax=ax[3])
    plt.show()


def show_mfcc(y, n_mfcc=20, figsize=(10,3)):
    mfccs = get_mfcc(y, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=f'MFCC {n_mfcc}')
    ax.set_yticks(range(0, n_mfcc), labels=range(1, n_mfcc + 1))
    plt.show()


def show_chromagram(y, sr=22050, hop_length=512, method='stft', cmap='gray_r', title='Chromagram', figsize=(10,3)):
    chromagram = get_chromagram(y, sr=sr, hop_length=hop_length, method=method)

    f, ax = plt.subplots(figsize=figsize)
    ax.set(title=title+f" {method.upper()}")
    img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', cmap=cmap, ax=ax)
    f.colorbar(img, ax=ax)
    plt.show()


def show_tonnetz(y, sr=22050, hop_length=512, method='cqt', cmap='gray_r', figsize=(10,6)):
    tonnetz = get_tonnetz(y, sr=sr)
    chromagram = get_chromagram(y, sr=sr, hop_length=hop_length, method=method)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    img1 = librosa.display.specshow(tonnetz,
                                    y_axis='tonnetz', x_axis='time', ax=ax[0])
    ax[0].set(title='Tonal Centroids (Tonnetz)')
    ax[0].label_outer()
    img2 = librosa.display.specshow(chromagram,
                                    y_axis='chroma', x_axis='time', ax=ax[1], cmap=cmap)
    ax[1].set(title=f'Chromagram {method.upper()}')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])
    plt.show()


def show_zcr(y, sr=22050, figsize=(10,6)):
    zcr = get_zero_crossing_rate(y=y)
    times = librosa.times_like(zcr)
    fig, ax = plt.subplots(2, sharex=True, figsize=figsize)
    ax[0].plot(times, zcr, c='c')
    ax[0].legend(['Zero-crossing rate'])
    librosa.display.waveshow(y=y, sr=sr, ax=ax[1])
    ax[1].set_xlabel('Time')
    plt.show()


def show_rms(y, sr=22050, figsize=(10,6)):
    rms = get_rms(y)
    times = librosa.times_like(rms)
    fig, ax = plt.subplots(2, sharex=True, figsize=figsize)
    ax[0].plot(times, rms, label='RMS Energy', c='c')
    ax[0].legend()
    librosa.display.waveshow(y=y, sr=sr, ax=ax[1])
    ax[1].set_xlabel('Time')
    plt.show()


def show_tempogram(y, sr=22050, hop_length=512, figsize=(10,10)):
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)

    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)

    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr)[0]
    fig, ax = plt.subplots(nrows=3, figsize=figsize)

    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[0])
    ax[0].axhline(tempo, color='w', linestyle='--', alpha=1,
                  label='Estimated tempo={:g}'.format(tempo))
    ax[0].legend(loc='upper right')
    ax[0].set(title='Tempogram')

    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                    num=tempogram.shape[0])
    ax[1].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    ax[1].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    ax[1].set(xlabel='Lag (seconds)')
    ax[1].legend()

    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    ax[2].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                   label='Mean local autocorrelation', base=2)
    ax[2].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                   label='Global autocorrelation', base=2)
    ax[2].axvline(tempo, color='black', linestyle='--', alpha=.8,
                  label='Estimated tempo={:g}'.format(tempo))
    ax[2].legend()
    ax[2].set(xlabel='BPM')
    ax[2].grid(True)
    plt.show()


def show_onset_strength(y, sr=22050, figsize=(10,6)):
    onset_env = get_onset_strength(y, sr=sr)
    fig, ax = plt.subplots(nrows=2, figsize=figsize)
    times = librosa.times_like(onset_env, sr=sr)
    ax[0].plot(times, onset_env, label='Onset strength', c='y')
    ax[0].label_outer()
    ax[0].legend()
    librosa.display.waveshow(y=y, sr=sr, x_axis='time', ax=ax[1])
    plt.show()


def show_bpm(y, sr=22050, start_bpm=100, figsize=(10,3), listen=True):
    bpm, beats = get_bpm(y, start_bpm=start_bpm, units='time', return_beats=True)
    f, ax = plt.subplots(figsize=figsize)
    ax.set(title=f'Beats (BPM={bpm})')
    librosa.display.waveshow(y=y, sr=sr, ax=ax)
    ax.vlines(beats, ymin=-y.max() * 1.1, ymax=y.max() * 1.1, color='orange')
    plt.show()

    if listen is True:
        clicks = librosa.clicks(times=beats, sr=sr, length=len(y))
        ipd.display(ipd.Audio(y + clicks, rate=sr))
    return bpm