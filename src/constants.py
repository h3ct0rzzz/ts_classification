import datetime
import numpy as np

EPOCH_DATE = datetime.date(1970, 1, 1)
BENFORD_DIST = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])

QUANTILE_25_VALUE = 0.25
QUANTILE_75_VALUE = 0.75
AUTOCORRELATION_LAG = 1
FFT_COEFFICIENT_0 = 0
FFT_COEFFICIENT_1 = 1
NUMBER_CROSSING_VALUE = 0

