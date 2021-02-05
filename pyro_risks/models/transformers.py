from typing import List, Union, Optional, Dict, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils import column_or_1d
from datetime import datetime

import pandas as pd
import numpy as np


class TargetDiscretizer:
    """Discretize numerical target variable.

    The `TargetDiscretizer` transformer maps target variable values to discrete values using
    a user defined function.

    Parameters:
        discretizer: user defined function.
    """

    def __init__(self, discretizer: callable):

        if callable(discretizer):
            self.discretizer = discretizer
        else:
            raise TypeError(
                f'{self.__class__.__name__} constructor expect a callable')

    def fit_resample(self, X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Discretize the target variable.

        The `fit_resample` method allows for discretizing the target variable.
        The method does not resample the dataset, the naming convention ensure
        the compatibility of the transformer with imbalanced-learn `Pipeline`
        object.

        Args:
            X: Training dataset features
            y: Training dataset target

        Returns:
                Training dataset features and target tuple.
        """

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            X = X.copy()
            y = y.copy()

        else:
            raise TypeError(
                f'{self.__class__.__name__} fit_resample methods expect pd.DataFrame and\
                    pd.Series as inputs.')

        y = y.apply(self.discretizer)

        return X, y


class CategorySelector:
    """Select features and targets rows.

        The `CategorySelector` transformer select features and targets rows
        belonging to given variable categories.

        Parameters:
            variable: variable to be used for selection.
            category: modalities to be selected.
        """

    def __init__(self, variable: str, category: Union[str, list]):

        self.variable = variable
        # Catch or prevent key errors
        if isinstance(category, str):
            self.category = [category]
        elif isinstance(category, list):
            self.category = category
        else:
            raise TypeError(
                f'{self.__class__.__name__} constructor category argument expect a string or a list'
            )

    def fit_resample(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Select features and targets rows.

        The `fit_resample` method allows for selecting the features and target
        rows. The method does not resample the dataset, the naming convention ensure
        the compatibility of the transformer with imbalanced-learn `Pipeline`
        object.

        Args:
            X: Training dataset features
            y: Training dataset target

        Returns:
                Training dataset features and target tuple.
        """

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            mask = X[self.variable].isin(self.category)
            XR = X[mask].copy()
            yr = y[mask].copy()

        else:
            raise TypeError(
                f'{self.__class__.__name__} fit_resample methods expect pd.DataFrame and\
                    pd.Series as inputs.')

        return XR, yr


class Imputer(SimpleImputer):
    """Impute missing values.

    The `Imputer` transformer wraps scikit-learn SimpleImputer transformer.

    Parameters:
        missing_values: the placeholder for the missing values.
        strategy: the imputation strategy (mean, median, most_frequent, constant).
        fill_value: fill_value is used to replace all occurrences of missing_values (default to 0).
        verbose: controls the verbosity of the imputer.
        copy: If True, a copy of X will be created.
        add_indicator: If True, a MissingIndicator transform will stack onto output of the imputer’s transform.
    """

    def __init__(self,
                 missing_values: Union[int, float, str] = np.nan,
                 strategy: str = 'mean',
                 fill_value: float = None,
                 verbose: int = 0,
                 copy: bool = True,
                 add_indicator: bool = False):
        super().__init__(missing_values=missing_values,
                         strategy=strategy,
                         fill_value=fill_value,
                         verbose=verbose,
                         copy=copy,
                         add_indicator=add_indicator)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the imputer on X.

        Args:
            X: Training dataset features.
            y: Training dataset target.

        Returns:
                Transformer.
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            X = X.copy()
            y = y.copy()
        else:
            raise TypeError(
                f'{self.__class__.__name__} transformer fit methods expect pd.DataFrame\
                    and pd.Series as inputs.')
        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute X missing values.

        Args:
            X: Training dataset features.

        Returns:
                Transformed training dataset.
        """
        X[X.columns] = super().transform(X)

        return X
