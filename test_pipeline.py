"""Data contract and preprocessing checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.train import FEATURES, ROOT, load_data, make_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(ROOT / "data/heart_disease_midterm.csv")

    def load_frame(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            frame.to_csv(path, index=False)
            return load_data(path)

    def test_duplicate_removal_and_target_encoding(self):
        X, y, audit = self.load_frame(pd.concat([self.raw, self.raw.iloc[[0]]]))
        self.assertEqual(len(X), len(self.raw.drop_duplicates()))
        self.assertEqual(audit["duplicate_rows_removed"], 2)
        self.assertEqual(y.iloc[0], 0)
        self.assertEqual(set(y), {0, 1})

    def test_invalid_target_is_rejected(self):
        self.raw.loc[0, "disease"] = "unknown"
        with self.assertRaises(ValueError):
            self.load_frame(self.raw)

    def test_conflicting_labels_are_rejected(self):
        other = self.raw.iloc[[0]].copy()
        other["disease"] = "yes"
        with self.assertRaises(ValueError):
            self.load_frame(pd.concat([self.raw, other]))

    def test_missing_predictors_are_imputed_and_test_data_does_not_refit(self):
        X, y, _ = self.load_frame(self.raw)
        train = X.iloc[:220].copy()
        train.iloc[0, 0] = np.nan
        model = make_pipeline(LogisticRegression(max_iter=2000)).set_params(select__k="all")
        model.fit(train, y.iloc[:220])
        scaler = model["preprocess"].named_transformers_["numeric"]["scale"]
        before = scaler.mean_.copy()
        test = X.iloc[220:].copy()
        test["age"] = 1000
        test["chol"] = np.nan
        probabilities = model.predict_proba(test)
        np.testing.assert_array_equal(scaler.mean_, before)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(len(model["preprocess"].get_feature_names_out()), len(FEATURES))


if __name__ == "__main__":
    unittest.main()
