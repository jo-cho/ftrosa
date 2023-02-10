import numpy as np
import pandas as pd


def get_stats_from_df(df, stats=None):
    cols_ = ['mean', 'std', 'skew', 'kurt', 'max', 'min']

    out_df = pd.concat([df.mean(), df.std(), df.skew(), df.kurt(),
                        df.max(), df.min()], axis=1)
    out_df.columns = cols_
    if stats is not None:
        out_df = out_df[stats]
    return out_df


def get_feature_stats(df, stats=None, song_name='song_name'):
    stats_df = get_stats_from_df(df=df, stats=stats)
    x_list = []
    x_name_list = []
    for i in stats_df.index:
        for j in stats_df.columns:
            x_ij = stats_df.loc[i, j]
            x_ij_name = i+'_'+j
            x_list.append(x_ij)
            x_name_list.append(x_ij_name)
    out = pd.DataFrame(dict(zip(x_name_list, x_list)), index=[song_name]).T
    return out
