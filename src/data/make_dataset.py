import pandas as pd
import numpy as np
from src.features.build_features import *
from constants import (
    QUANTILE_25_VALUE,
    QUANTILE_75_VALUE,
    AUTOCORRELATION_LAG,
    FFT_COEFFICIENT_0,
    FFT_COEFFICIENT_1,
    NUMBER_CROSSING_VALUE
)


def remove_rows_with_null_values(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['values'].apply(lambda x: any(pd.isnull(v) for v in x))]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_dict = {
        'id': df['id'],
        'start_date': df['dates'].apply(get_start_date),
        'end_date': df['dates'].apply(get_end_date),
        'duration': df['dates'].apply(calculate_duration),
    }

    for name, func in globals().items():
        if hasattr(func, 'is_feature'):
            if name == 'quantile':
                feature_dict['quantile_25'] = df['values'].apply(lambda x: func(x, QUANTILE_25_VALUE))
                feature_dict['quantile_75'] = df['values'].apply(lambda x: func(x, QUANTILE_75_VALUE))
            elif name == 'autocorrelation':
                feature_dict['autocorrelation_lag_1'] = df['values'].apply(lambda x: func(x, AUTOCORRELATION_LAG))
            elif name == 'fft_coefficient':
                feature_dict['fft_coefficient_0'] = df['values'].apply(lambda x: np.abs(func(x, FFT_COEFFICIENT_0)))
                feature_dict['fft_coefficient_1'] = df['values'].apply(lambda x: np.abs(func(x, FFT_COEFFICIENT_1)))
            elif name == 'number_crossing_m':
                feature_dict['number_crossing_0'] = df['values'].apply(lambda x: func(x, NUMBER_CROSSING_VALUE))
            else:
                feature_dict[name] = df['values'].apply(func)

    processed_df = pd.DataFrame(feature_dict)

    if 'label' in df.columns:
        processed_df['y'] = df['label']

    return processed_df


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = remove_rows_with_null_values(df)
    return generate_features(df_clean)
