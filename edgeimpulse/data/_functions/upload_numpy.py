# mypy: ignore-errors
# ruff: noqa: D100
from edgeimpulse.data.sample_type import (
    Sample,
    DataAcquisition,
    Sensor,
    Protected,
    Payload,
    UploadSamplesResponse,
)
from edgeimpulse.data._functions.upload import (
    upload_samples,
)
from typing import Optional, Literal, List
import random
import json
from dataclasses import asdict

DEVICE_TYPE = "EDGE_IMPULSE_PYTHON_SDK"


def upload_numpy(
    data,
    labels: List[str],
    sensors: List[Sensor],
    sample_rate_ms: int,
    metadata: Optional[dict] = None,
    category: Literal["training", "testing", "split", "anomaly"] = "split",
) -> UploadSamplesResponse:
    """Upload numpy arrays as timeseries using the Edge Impulse data acquisition format.

    Args:
        data (array): Numpy array containing the timeseries data. The shape should be (num_samples, time_point, num_sensors)
        labels (List[str]): List of labels for the data samples can also be a numpy array.
        sensors (List[Sensor]): List of Sensor objects representing the sensors used in the data.
        sample_rate_ms (int): Time interval in milliseconds between consecutive data points.
        metadata (dict, optional): Metadata for all samples being uploaded. Default is None.
        category (str or None, optional): Category or class label for the entire dataset. Default is split.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        ValueError: If the length of labels doesn't match the number of samples or if the number of sensors
            doesn't match the number of axes in the data.

    Examples:
        Uploads numpy data

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        import numpy as np
        from edgeimpulse import data

        # Create 2 samples, each with 3 axes of accelerometer data
        values = np.array(
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
            ]
        )

        # The labels for each sample
        labels = ["up", "down"]

        # The sensors used in the samples
        sensors = [
            {"name": "accelX", "units": "ms/s"},
            {"name": "accelY", "units": "ms/s"},
            {"name": "accelZ", "units": "ms/s"},
        ]

        # Upload samples to your Edge Impulse project
        resp = data.upload_numpy(
            sample_rate_ms=100,
            data=values,
            labels=labels,
            category="training",
            sensors=sensors,
        )
        print(resp)
        ```
    """
    if len(data) != len(labels):
        raise ValueError(
            f"Labels length ({len(labels)}) must be equal to the number of samples given ({len(data)})"
        )

    if len(data[0][0]) != len(sensors):
        raise ValueError(
            f"Number of sensors ({len(sensors)}) doesn't match the number of axes in the data ({len(data[0][0])})"
        )

    samples = []
    for i, d in enumerate(data):
        values = d.tolist() if hasattr(d, "tolist") else d
        sample = numpy_timeseries_to_sample(
            sample_rate_ms=sample_rate_ms, values=values, sensors=sensors
        )
        sample.label = labels[i]
        sample.metadata = metadata
        sample.category = category
        samples.append(sample)

    return upload_samples(samples)


def numpy_timeseries_to_sample(
    values, sensors: List[Sensor], sample_rate_ms: int
) -> Sample:
    """Convert numpy values to a sample that can be uploaded to Edge Impulse.

    Args:
        values (array): Numpy array containing the timeseries data. The shape should be (num_samples, time_point, num_sensors)
        sensors (List[Sensor]): List of sensor objects representing the sensors used in the data.
        sample_rate_ms (int): Time interval in milliseconds between consecutive data points.

    Returns:
        Sample: Sample object that can be uploaded to Edge Impulse
    """
    data = DataAcquisition(
        protected=Protected(),
        payload=Payload(
            device_type=DEVICE_TYPE,
            interval_ms=sample_rate_ms,
            values=values,
            sensors=sensors,
        ),
    )
    sample = Sample(
        filename="%08x.json" % random.getrandbits(64), data=json.dumps(asdict(data))
    )
    return sample
