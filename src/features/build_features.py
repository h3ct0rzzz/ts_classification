import datetime
import itertools
from functools import wraps
from typing import Union, List

import numpy as np
import pandas as pd
from scipy.stats import linregress, entropy
from src.constants import EPOCH_DATE, BENFORD_DIST


def feature_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    setattr(wrapper, 'is_feature', True)
    return wrapper


@feature_function
def linear_trend(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    time_index = np.arange(len(data))
    slope, _, _, _, _ = linregress(time_index, data)
    return slope


@feature_function
def phase_duration(data: Union[List[float], np.ndarray]) -> int:
    data = np.asarray(data)
    mean_value = np.mean(data)
    above_mean = data > mean_value
    max_phase = current_phase = 1

    for i in range(1, len(above_mean)):
        if above_mean[i] == above_mean[i - 1]:
            current_phase += 1
        else:
            max_phase = max(max_phase, current_phase)
            current_phase = 1

    return max(max_phase, current_phase)


@feature_function
def max_trend_length(data: Union[List[float], np.ndarray]) -> int:
    data = np.asarray(data)
    diffs = np.diff(data)
    trend_length = max_trend = 1

    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i-1] > 0:
            trend_length += 1
        else:
            max_trend = max(max_trend, trend_length)
            trend_length = 1

    return max(max_trend, trend_length)


@feature_function
def time_series_entropy(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    value_counts = np.histogram(data, bins=10, density=True)[0]
    return entropy(value_counts + 1e-10)


@feature_function
def calculate_iqr(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.percentile(data, 75) - np.percentile(data, 25)


def calculate_date(date: datetime.date) -> int:
    return (date - EPOCH_DATE).days


@feature_function
def get_start_date(dates: List[datetime.date]) -> int:
    return calculate_date(dates[0])


@feature_function
def get_end_date(dates: List[datetime.date]) -> int:
    return calculate_date(dates[-1])


@feature_function
def calculate_duration(dates: List[datetime.date]) -> int:
    return get_end_date(dates) - get_start_date(dates)


@feature_function
def variance_larger_than_standard_deviation(data: Union[List[float], np.ndarray]) -> bool:
    data = np.asarray(data)
    std = np.std(data)
    return std * std > std


@feature_function
def has_duplicate_max(data: Union[List[float], np.ndarray]) -> bool:
    data = np.asarray(data)
    return np.sum(data == np.max(data)) >= 2


@feature_function
def has_duplicate_min(data: Union[List[float], np.ndarray]) -> bool:
    data = np.asarray(data)
    return np.sum(data == np.min(data)) >= 2


@feature_function
def has_duplicates(data: Union[List[float], np.ndarray]) -> bool:
    data = np.asarray(data)
    return data.size != np.unique(data).size


@feature_function
def sum_values(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.sum(data)


@feature_function
def abs_energy(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.dot(data, data)


@feature_function
def complexity_invariant_distance(data: Union[List[float], np.ndarray], normalize: bool = False) -> float:
    data = np.asarray(data)
    data_copy = data.copy()
    if normalize:
        std = np.std(data_copy)
        if std != 0:
            data_copy = (data_copy - np.mean(data_copy)) / std
        else:
            return 0.0
    return np.sqrt(np.sum(np.diff(data_copy)**2))


@feature_function
def mean_abs_change(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.mean(np.abs(np.diff(data)))


@feature_function
def mean_change(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return (data[-1] - data[0]) / (len(data) - 1) if len(data) > 1 else np.nan


@feature_function
def mean_second_derivative_central(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return (data[-1] - data[-2] - data[1] + data[0]) / (2 * (len(data) - 2)) if len(data) > 2 else np.nan


@feature_function
def median(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.median(data)


@feature_function
def mean(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.mean(data)


@feature_function
def length(data: Union[List[float], np.ndarray]) -> int:
    data = np.asarray(data)
    return len(data)


@feature_function
def standard_deviation(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.std(data)


@feature_function
def variation_coefficient(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    data_mean = np.mean(data)
    return np.std(data) / data_mean if data_mean != 0 else np.nan


@feature_function
def variance(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.var(data)


@feature_function
def skewness(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return pd.Series(data).skew(skipna=False)


@feature_function
def kurtosis(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return pd.Series(data).kurtosis()


@feature_function
def root_mean_square(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.sqrt(np.mean(np.square(data))) if len(data) > 0 else np.nan


@feature_function
def absolute_sum_of_changes(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.sum(np.abs(np.diff(data)))


@feature_function
def count_above_mean(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.sum(data > np.mean(data))


@feature_function
def count_below_mean(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.sum(data < np.mean(data))


@feature_function
def last_location_of_maximum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return 1.0 - np.argmax(data[::-1]) / len(data) if len(data) > 0 else np.nan


@feature_function
def first_location_of_maximum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.argmax(data) / len(data) if len(data) > 0 else np.nan


@feature_function
def last_location_of_minimum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return 1.0 - np.argmin(data[::-1]) / len(data) if len(data) > 0 else np.nan


@feature_function
def first_location_of_minimum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.argmin(data) / len(data) if len(data) > 0 else np.nan


@feature_function
def percentage_of_reoccurring_values(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    if len(data) == 0:
        return np.nan
    _, counts = np.unique(data, return_counts=True)
    return np.sum(counts > 1) / float(counts.shape[0]) if counts.shape[0] > 0 else 0


@feature_function
def percentage_of_reoccurring_datapoints(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    if len(data) == 0:
        return np.nan
    value_counts = pd.Series(data).value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    return reoccuring_values / len(data) if not np.isnan(reoccuring_values) else 0


@feature_function
def sum_of_reoccurring_values(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    unique, counts = np.unique(data, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


@feature_function
def ratio_unique_values(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.unique(data).size / data.size if data.size > 0 else np.nan


@feature_function
def quantile(data: Union[List[float], np.ndarray], quantile_value: float) -> float:
    data = np.asarray(data)
    return np.quantile(data, quantile_value) if len(data) > 0 else np.nan


@feature_function
def maximum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.max(data)

@feature_function
def absolute_maximum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.max(np.absolute(data)) if len(data) > 0 else np.nan


@feature_function
def minimum(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    return np.min(data)


@feature_function
def benford_correlation(data: Union[List[float], np.ndarray]) -> float:
    data = np.asarray(data)
    first_digits = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(data))])
    data_dist = np.array([(first_digits == n).mean() for n in range(1, 10)])
    return np.corrcoef(BENFORD_DIST, data_dist)[0, 1]


@feature_function
def autocorrelation(data: Union[List[float], np.ndarray], lag: int) -> float:
    data = np.asarray(data)
    return np.corrcoef(data[lag:], data[:-lag])[0, 1] if len(data) > lag else np.nan


@feature_function
def fft_coefficient(data: Union[List[float], np.ndarray], coefficient: int = 0) -> complex:
    data = np.asarray(data)
    fft = np.fft.fft(data)
    return fft[coefficient] if len(fft) > coefficient else np.nan


@feature_function
def number_crossing_m(data: Union[List[float], np.ndarray], threshold: float) -> int:
    data = np.asarray(data)
    greater = data > threshold
    return np.sum(np.diff(greater.astype(int)) != 0)


@feature_function
def energy_ratio_by_chunks(data: Union[List[float], np.ndarray], num_segments: int = 10, segment_focus: int = 0) -> float:
    data = np.asarray(data)
    data_abs = np.abs(data)
    segments = np.array_split(data_abs, num_segments)
    segment_energy = np.sum(segments[segment_focus] ** 2)
    total_energy = np.sum(data_abs ** 2)
    return segment_energy / total_energy if total_energy != 0 else np.nan


@feature_function
def permutation_entropy(data: Union[List[float], np.ndarray], order: int = 3, delay: int = 1) -> float:
    data = np.asarray(data)
    permutations = np.array(list(itertools.permutations(range(order))))
    counts = [0] * len(permutations)

    for i in range(len(data) - delay * (order - 1)):
        sorted_idx = np.argsort(data[i:i + delay * order:delay])
        for j, perm in enumerate(permutations):
            if np.all(sorted_idx == perm):
                counts[j] += 1
                break

    counts = np.array(counts) / float(sum(counts))
    return -np.sum(counts[counts > 0] * np.log(counts[counts > 0])) / np.log(float(len(permutations)))
