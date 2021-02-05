import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyro_risks.models import TargetDiscretizer


class TransformersTester(unittest.TestCase):

    def test_target_discretizer(self):
        td = TargetDiscretizer(discretizer=lambda x: 1 if x > 0 else 0)
        df = pd.DataFrame({
            "day": ["2019-07-01", "2019-08-02", "2019-06-12"],
            "departement": ["Aisne", "Cantal", "Savoie"],
            "fires": [0, 5, 10],
            "fwi_mean": [13.3, 0.9, 2.5],
            "ffmc_max": [23, 45.3, 109.0],
        })
        X = df.drop(columns=['fires'])
        y = df['fires']

        Xr, yr = td.fit_resample(X, y)
        assert_series_equal(yr, pd.Series([0, 1, 1], name="fires"))
        assert_frame_equal(Xr, X)
        self.assertRaises(TypeError, TargetDiscretizer, [0, 1])
        self.assertRaises(TypeError, TargetDiscretizer.fit_resample,
                          np.array([[0, 0, 0], [0, 0, 0]]), np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
