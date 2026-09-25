import math

import numpy as np
import pandas as pd


def to_native(value):
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else round(number, 6)
    if isinstance(value, pd.Period):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
