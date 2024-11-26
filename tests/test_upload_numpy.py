# ruff: noqa: D100, D101, D102, D103
import unittest
import logging
from tests.util import delete_all_samples, assert_uploaded_samples
from edgeimpulse import data
from edgeimpulse import util

logging.getLogger().setLevel(logging.INFO)


# ruff: noqa: F821
def load_timeseries() -> "np.array":  # type: ignore
    """Load the timeseries dataset."""
    import numpy as np

    # create 5 samples, with 3 axis (sensors)
    samples = np.array(
        [
            [  # sample 1
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.12, 0.03, 1.23],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 2
                [8.81, 0.03, 1.21],
                [9.12, 0.03, 1.23],
                [9.14, 2.01, 1.25],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 3
                [8.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 4
                [9.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 5
                [10.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
        ]
    )

    sensors = [
        {"name": "accelX", "units": "ms/s"},
        {"name": "accelY", "units": "ms/s"},
        {"name": "accelZ", "units": "ms/s"},
    ]

    labels = ["up", "down", "down", "up", "down"]
    return (samples, labels, sensors)


class TestUploadNumpy(unittest.TestCase):
    @unittest.skipUnless(
        util.numpy_installed(), "Test requires numpy but it was not available"
    )
    def setUp(self):
        delete_all_samples()

    def test_upload_single_data(self):
        sample = data.numpy_timeseries_to_sample(
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

        res = data.upload_samples(sample, allow_duplicates=True)
        self.assertEqual(len(res.fails), 0)
        self.assertEqual(len(res.successes), 1)

        assert_uploaded_samples(self, res.successes)

    def test_upload_numpy_timeseries_data(self):
        (samples, labels, sensors) = load_timeseries()

        res = data.upload_numpy(
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

        (samples, labels, sensors) = load_timeseries()

        with self.assertRaises(ValueError) as context:
            samples = np.delete(samples, 2)
            data.upload_numpy(
                sample_rate_ms=100,
                data=samples,
                labels=labels,
                category="training",
                sensors=sensors,
            )
        self.assertIn(
            "Labels length (5) must be equal to the number of samples given (59)",
            str(context.exception),
        )

    def test_incorrect_axis(self):
        (samples, labels, sensors) = load_timeseries()

        with self.assertRaises(ValueError) as context:
            sensors.append({"name": "accelZ", "units": "ms/s"})
            data.upload_numpy(
                sample_rate_ms=100,
                data=samples,
                labels=labels,
                category="training",
                sensors=sensors,
            )
        self.assertIn(
            "Number of sensors (4) doesn't match the number of axes in the data (3)",
            str(context.exception),
        )
