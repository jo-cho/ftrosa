import numpy as np
import librosa


def get_y_from_audio(path_audio, sr=22050, duration=30, start=10, trim=True):
    y, sr = librosa.load(path_audio, sr=sr, duration=duration, offset=start)
    if trim is True:
        y = librosa.effects.trim(y)[0]  # check if there is silence before or after the actual audio
    return y


def hpss(y, margin=1.0):
    """
    Median-filtering harmonic percussive source separation (HPSS).

    If ``margin = 1.0``, decomposes an input spectrogram ``S = H + P``
    where ``H`` contains the harmonic components,
    and ``P`` contains the percussive components.

    If ``margin > 1.0``, decomposes an input spectrogram ``S = H + P + R``
    where ``R`` contains residual components not included in ``H`` or ``P``.
    """
    y_harm, y_perc = librosa.effects.hpss(y=y, margin=margin)
    out = (y_harm, y_perc)
    return out


def stft_mag(y, n_fft=2048, hop_length=None, window='hann', center=True,
             power='energy', ref=np.max,
             return_dB=False):
    # STFT
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center)  # stft coefficient

    # Amplitude
    if power == 'energy':
        S, P = librosa.magphase(D, power=1)  # S = np.abs(D) energy
    elif power == 'power':
        S, P = librosa.magphase(D, power=2)  # S = np.abs(D)**2 power
    else:
        raise Exception("power must be either ['energy', 'power]")
    out = S
    # Convert an amplitude spectrogram to dB-scaled spectrogram
    S_dB = librosa.amplitude_to_db(S, ref=ref)
    if return_dB is True:
        out = S_dB
    return out


def get_spectral_centroids(y, sr=22050):
    """
    Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.
    """
    # Calculate the Spectral Centroids
    spec_centr = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return spec_centr


def get_spectral_bandwidth(y, sr=22050, p=2):
    """
    Compute p'th-order spectral bandwidth.
    """
    # Calculate the Spectral Centroids
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=p)[0]
    return spec_bw


def get_spectral_contrast(y, sr=22050,
                          n_bands=6, quantile=0.02):
    """
    Compute spectral contrast

    Each frame of a spectrogram ``S`` is divided into sub-bands.
    For each sub-band, the energy contrast is estimated by comparing
    the mean energy in the top quantile (peak energy) to that of the
    bottom quantile (valley energy).

    n_bands : int > 1
        number of frequency bands

    Returns
    -------
    contrast : np.ndarray [shape=(..., n_bands + 1, t)]
        each row of spectral contrast values corresponds to a given
        octave-based frequency
    """
    # Calculate the Spectral Centroids
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, quantile=quantile)
    return spec_contrast


def get_spectral_flatness(y, n_fft=2048, hop_length=512):
    """
    Compute spectral flatness

    Spectral flatness (or tonality coefficient) is a measure to
    quantify how much noise-like a sound is, as opposed to being
    tone-like. A high spectral flatness (closer to 1.0)
    indicates the spectrum is similar to white noise.
    """
    # Calculate the Spectral Centroids
    spec_flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    return spec_flat


def get_spectral_rolloff(y, sr=22050, roll_percent=0.85):
    """Compute roll-off frequency.

    The roll-off frequency is defined for each frame as the center frequency
    for a spectrogram bin such that at least roll_percent (0.85 by default)
    of the energy of the spectrum in this frame is contained in this bin and
    the bins below.
    """
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    return spec_rolloff


def get_spectral_poly(y, sr=22050, order=1):
    """
    Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.
    """
    spec_poly = librosa.feature.poly_features(y=y, sr=sr, order=order)
    return spec_poly


def get_mfcc(y, sr=22050, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def get_chromagram(y, sr=22050, hop_length=512, n_chroma=12, method='stft'):
    if method == 'stft':
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    elif method == 'cqt':
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    elif method == 'cens':
        chromagram = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    else:
        raise Exception("method: ['stft','cqt','cens']")
    return chromagram


def get_tonnetz(y, sr=22050):
    """
    Computes the tonal centroid features (tonnetz)

    """
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return tonnetz


def get_zero_crossing_rate(y, frame_length=2048, hop_length=512):
    """
    Compute the zero-crossing rate of an audio time series
    """
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return zcr


def get_rms(y, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def get_onset_strength(y, sr=22050):
    onset_str = librosa.onset.onset_strength(y=y, sr=sr)
    return onset_str


def get_bpm(y_perc, sr=22050, start_bpm=100, units='time', return_beats=False):
    tempo, beats = librosa.beat.beat_track(y=y_perc, sr=sr, start_bpm=start_bpm, units=units)
    out = tempo
    if return_beats is True:
        out = (tempo, beats)
    return out
