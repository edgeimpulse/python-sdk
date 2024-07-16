# ruff: noqa: D100, D101, D102, D103
import unittest
from edgeimpulse import util
import logging
from tests.util import delete_all_samples, assert_uploaded_samples
import edgeimpulse as ei

logging.getLogger().setLevel(logging.INFO)


class TestUploadNumpy(unittest.TestCase):
    @unittest.skipUnless(
        util.numpy_installed(), "Test requires numpy but it was not available"
    )
    def setUp(self):
        delete_all_samples()

    def test_upload_single_data(self):
        sample = ei.experimental.data.numpy_timeseries_to_sample(
            values=[
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.12, 0.03, 1.23],
                [9.14, 2.01, 1.25],
            ],
            sensors=[
                {"name": "accelX", "units": "ms/s"},
                {"name": "accelY", "units": "ms/s"},
                {"name": "accelZ", "units": "ms/s"},
            ],
            sample_rate_ms=100,
        )

        res = ei.experimental.data.upload_samples(sample, allow_duplicates=True)
        self.assertEqual(len(res.fails), 0)
        self.assertEqual(len(res.successes), 1)

        assert_uploaded_samples(self, res.successes)

    def test_upload_numpy_timeseries_data(self):
        (samples, labels, sensors) = ei.datasets.load_timeseries()

        res = ei.experimental.data.upload_numpy(
            sample_rate_ms=100,
            data=samples,
            labels=labels,
            category="training",
            sensors=sensors,
        )

        self.assertEqual(len(res.fails), 0)
        self.assertEqual(len(res.successes), 5)

        assert_uploaded_samples(self, res.successes)

    def test_incorrect_numpy_labels(self):
        import numpy as np

        (samples, labels, sensors) = ei.datasets.load_timeseries()

        with self.assertRaises(ValueError) as context:
            samples = np.delete(samples, 2)
            ei.experimental.data.upload_numpy(
                sample_rate_ms=100,
                data=samples,
                labels=labels,
                category="training",
                sensors=sensors,
            )
        self.assertIn(
            "Labels length (5) must be equal length of samples given (59)",
            str(context.exception),
        )

    def test_incorrect_axis(self):
        (samples, labels, sensors) = ei.datasets.load_timeseries()

        with self.assertRaises(ValueError) as context:
            sensors.append({"name": "accelZ", "units": "ms/s"})
            ei.experimental.data.upload_numpy(
                sample_rate_ms=100,
                data=samples,
                labels=labels,
                category="training",
                sensors=sensors,
            )
        self.assertIn(
            "Number of sensors (4) doesn't match number of axis in the data (3)",
            str(context.exception),
        )
