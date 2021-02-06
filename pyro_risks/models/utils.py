from typing import List, Union, Optional, Dict, Tuple
import pandas as pd
import numpy as np


def check_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X = X.copy()
        y = y.copy()
    else:
        raise TypeError('Transformer methods expect pd.DataFrame\
                and pd.Series as inputs.')
    return X, y


def check_x(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X = X.copy()
    else:
        raise TypeError('Transformer methods expect pd.DataFrame as inputs')
    return X
