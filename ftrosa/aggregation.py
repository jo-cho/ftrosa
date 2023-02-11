from .features import *
from .feature_stats import *


def get_df_spec_features(y, sr=22050, n_fft=2048, hop_length=512, n_contrast_bands=4):
    """
    :param y:
    :param sr:
    :param n_fft:
    :param hop_length:
    :param n_contrast_bands: the number of spectral contrast sub-bands
    :return: (DataFrame)
    """
    spec_centr = get_spectral_centroids(y=y, sr=sr)
    spec_bw = get_spectral_bandwidth(y=y, sr=sr)
    spec_rolloff_max = get_spectral_rolloff(y=y, sr=sr, roll_percent=.99)
    spec_rolloff_min = get_spectral_rolloff(y=y, sr=sr, roll_percent=.01)
    spec_flat = get_spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    spec_contrast = get_spectral_contrast(y=y, sr=sr, n_bands=n_contrast_bands)

    spec_features = [spec_centr, spec_bw, spec_rolloff_max, spec_rolloff_min, spec_flat]
    for i in range(n_contrast_bands+1):
        spec_features.append(spec_contrast[i])
    spec_features_columns = ['spectral_centroid', 'spectral_bandwidth',
                             'spectral_rolloff_max', 'spectral_rolloff_min',
                             'spectral_flatness']
    for i in range(1, n_contrast_bands+2):
        spec_features_columns.append(f'spectral_contrast_{i}')

    df_spec_feat = pd.DataFrame(np.array(spec_features).T, columns=spec_features_columns)
    return df_spec_feat


def get_df_mfcc(y, sr=22050, n_mfcc=20):
    """
    get_df_mfcc
    :param y:
    :param sr:
    :param n_mfcc: the number of MFCCs
    :return: (DataFrame)
    """
    mfccs = get_mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    df_mfcc = pd.DataFrame(np.array(mfccs).T, columns=[f"mfcc_{i}" for i in range(1, n_mfcc+1)])
    return df_mfcc


def get_df_chroma_features(y_harm, sr=22050, hop_length=512, method_list=['stft']):
    """

    :param y_harm: harmonic part of y (recommended)
    :param sr:
    :param hop_length:
    :param method_list: list of strings in ['stft','cqt','cens']
    :return: (DataFrame)
    """
    pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tonnetz_class = ['fifth_x', 'fifth_y', 'minor_x', 'minor_y', 'major_x', 'major_y']

    df_chroma_list = []
    for m in method_list:
        chromagram = get_chromagram(y=y_harm, sr=sr, hop_length=hop_length, method=m)
        df_chroma_ = pd.DataFrame(np.array(chromagram).T, columns=[f"chroma_{m}_{i}" for i in pitch_class])
        df_chroma_list.append(df_chroma_)
    df_chromagrams = pd.concat(df_chroma_list, axis=1)

    tonnetz = get_tonnetz(y=y_harm, sr=sr)
    df_tonnetz = pd.DataFrame(np.array(tonnetz.T), columns=[f'tonnetz_{i}' for i in tonnetz_class])

    df_chrom_feat = pd.concat([df_chromagrams, df_tonnetz], axis=1)
    return df_chrom_feat


def get_df_energy_features(y, sr=22050, frame_length=2048, hop_length=512):
    """
    get_df_energy_features
    :param y:
    :param sr:
    :param frame_length:
    :param hop_length:
    :return: (DataFrame)
    """
    zcr = get_zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    rms = get_rms(y=y, frame_length=frame_length, hop_length=hop_length)
    onset_str = get_onset_strength(y=y, sr=sr)
    df_energy_feat = pd.DataFrame({'zero_crossing_rate': zcr, 'rms': rms, 'onset_strength': onset_str})
    return df_energy_feat


def get_df_bpms(y_perc, sr=22050, start_bpms=[60, 90, 120], song_name='song_name'):
    """
    get_df_bpms
    :param y_perc:: percussive part of y (recommended)
    :param sr:
    :param start_bpms: (list) list of initial bpm for estimation
    :param song_name:
    :return: (DataFrame)
    """
    bpms = []
    for i in start_bpms:
        bpm_ = get_bpm(y_perc, sr=sr, start_bpm=i)
        bpms.append(bpm_)
    df_bpms = pd.DataFrame(np.array(bpms), index=[f'bpm_s{i}' for i in start_bpms], columns=[song_name])
    return df_bpms


def _get_all_raw_feats_from_y(y, y_harm, y_perc,
                              chroma_method_list=['stft', 'cqt', 'cens'],
                              n_contrast_bands=4, n_mfcc=12, start_bpms=[60, 120, 180],
                              chroma_harm=True, bpm_perc=True
                              ):
    raw_df_spec_feat = get_df_spec_features(y=y, n_contrast_bands=n_contrast_bands)
    raw_df_mfcc_feat = get_df_mfcc(y=y, n_mfcc=n_mfcc)
    if chroma_harm is False:
        raw_df_chroma_feat = get_df_chroma_features(y_harm=y, method_list=chroma_method_list)
    else:
        raw_df_chroma_feat = get_df_chroma_features(y_harm=y_harm, method_list=chroma_method_list)
    raw_df_energy_feat = get_df_energy_features(y=y)
    if bpm_perc is False:
        raw_df_bpm_feat = get_df_bpms(y_perc=y, start_bpms=start_bpms)
    else:
        raw_df_bpm_feat = get_df_bpms(y_perc=y_perc, start_bpms=start_bpms)
    out = (raw_df_spec_feat, raw_df_mfcc_feat, raw_df_chroma_feat, raw_df_energy_feat, raw_df_bpm_feat)
    return out


def _get_all_raw_sep_feats_from_y(y_harm, y_perc,
                                  chroma_method_list=['stft', 'cqt', 'cens'],
                                  n_contrast_bands=4, n_mfcc=12, start_bpms=[60, 120, 180],
                                  chroma_harm=True, bpm_perc=True
                                  ):
    raw_df_spec_feat_harm = get_df_spec_features(y_harm, n_contrast_bands=n_contrast_bands).rename(
        columns=lambda x: x + '_harm')
    raw_df_spec_feat_perc = get_df_spec_features(y_perc, n_contrast_bands=n_contrast_bands).rename(
        columns=lambda x: x + '_perc')
    raw_df_spec_feat = pd.concat([raw_df_spec_feat_harm, raw_df_spec_feat_perc], axis=1)

    raw_df_mfcc_feat_harm = get_df_mfcc(y_harm, n_mfcc=n_mfcc).rename(columns=lambda x: x + '_harm')
    raw_df_mfcc_feat_perc = get_df_mfcc(y_perc, n_mfcc=n_mfcc).rename(columns=lambda x: x + '_perc')
    raw_df_mfcc_feat = pd.concat([raw_df_mfcc_feat_harm, raw_df_mfcc_feat_perc], axis=1)

    if chroma_harm is False:
        raw_df_chroma_feat_harm = get_df_chroma_features(y_harm, method_list=chroma_method_list).rename(
            columns=lambda x: x + '_harm')
        raw_df_chroma_feat_perc = get_df_chroma_features(y_perc, method_list=chroma_method_list).rename(
            columns=lambda x: x + '_perc')
        raw_df_chroma_feat = pd.concat([raw_df_chroma_feat_harm, raw_df_chroma_feat_perc], axis=1)
    else:
        raw_df_chroma_feat = get_df_chroma_features(y_harm=y_harm, method_list=chroma_method_list)

    raw_df_energy_feat_harm = get_df_energy_features(y_harm).rename(columns=lambda x: x + '_harm')
    raw_df_energy_feat_perc = get_df_energy_features(y_perc).rename(columns=lambda x: x + '_perc')
    raw_df_energy_feat = pd.concat([raw_df_energy_feat_harm, raw_df_energy_feat_perc], axis=1)

    if bpm_perc is False:
        raw_df_bpm_feat_harm = get_df_bpms(y_harm, start_bpms=start_bpms).rename(
            columns=lambda x: x + '_harm')
        raw_df_bpm_feat_perc = get_df_bpms(y_perc, start_bpms=start_bpms).rename(
            columns=lambda x: x + '_perc')
        raw_df_bpm_feat = pd.concat([raw_df_bpm_feat_harm, raw_df_bpm_feat_perc], axis=1)
    else:
        raw_df_bpm_feat = get_df_bpms(y_perc=y_perc, start_bpms=start_bpms)
    out = (raw_df_spec_feat, raw_df_mfcc_feat, raw_df_chroma_feat, raw_df_energy_feat, raw_df_bpm_feat)
    return out


def _get_stats_from_raw_feats(_all_raw_feats, song_name, stats=None):
    raw_df_spec_feat, raw_df_mfcc_feat, raw_df_chroma_feat, raw_df_energy_feat, raw_df_bpm_feat = _all_raw_feats
    df_spec_feat = get_feature_stats(df=raw_df_spec_feat, stats=stats, song_name=song_name)
    df_mfcc_feat = get_feature_stats(raw_df_mfcc_feat, stats=stats, song_name=song_name)
    df_chroma_feat = get_feature_stats(raw_df_chroma_feat, stats=stats, song_name=song_name)
    df_energy_feat = get_feature_stats(raw_df_energy_feat, stats=stats, song_name=song_name)
    df_bpm_feat = raw_df_bpm_feat.rename(columns={'song_name': song_name})
    audio_features = pd.concat([df_spec_feat, df_mfcc_feat, df_energy_feat, df_chroma_feat, df_bpm_feat], axis=0)
    return audio_features


# overall
def get_all_musical_features(path_audio, song_name, stats=None,
                             duration=30, start=10,
                             from_harm_perc=False,
                             chroma_harm=True, bpm_perc=True,
                             sr=22050, hpr_margin=1.5,
                             chroma_method_list=['stft', 'cqt', 'cens'],
                             n_contrast_bands=4, n_mfcc=12, start_bpms=[60, 120, 180]):
    """
    Get all musical features
    :param path_audio: path of an audio file
    :param song_name: song name in string
    :param stats: default is None = ['mean','std','skew','kurt','max','min']
    :param from_harm_perc:
    :param chroma_harm:
    :param bpm_perc:
    :param hpr_margin:
    :param chroma_method_list:
    :param n_contrast_bands:
    :param n_mfcc:
    :param start_bpms:
    :return:
    """
    y = get_y_from_audio(path_audio, sr=sr, duration=duration, start=start)
    y_harm, y_perc = hpss(y=y, margin=hpr_margin)

    _all_raw_feats = _get_all_raw_feats_from_y(y, y_harm, y_perc,
                                               chroma_method_list=chroma_method_list,
                                               n_contrast_bands=n_contrast_bands, n_mfcc=n_mfcc, start_bpms=start_bpms,
                                               chroma_harm=chroma_harm, bpm_perc=bpm_perc)
    if from_harm_perc is True:
        _all_raw_feats = _get_all_raw_sep_feats_from_y(y_harm, y_perc,
                                                       chroma_method_list=chroma_method_list,
                                                       n_contrast_bands=n_contrast_bands, n_mfcc=n_mfcc,
                                                       start_bpms=start_bpms,
                                                       chroma_harm=chroma_harm, bpm_perc=bpm_perc)
    audio_features = _get_stats_from_raw_feats(_all_raw_feats, song_name, stats=stats)
    return audio_features
