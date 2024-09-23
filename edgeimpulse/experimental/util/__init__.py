"""This module contains all our experimental utilities.

Be warned these signatures may change.
"""

from edgeimpulse.util import numpy_installed, pandas_installed
from typing import Optional
import json

from edgeimpulse.experimental import data
import edgeimpulse as ei


def fetch_samples(
    filename: Optional[str] = None,
    category: Optional[str] = None,
    labels: Optional[str] = None,
    max_workers=None,
):
    """Fetch samples based on the provided parameters and stream them by their IDs.

    Args:
        filename (Optional[str]): The name of the file to fetch samples from. If None, samples are fetched from all files.
        category (Optional[str]): The category of samples to fetch. If None, samples from all categories are included.
        labels (Optional[str]): The labels associated with the samples to fetch. If None, samples with any labels are included.
        max_workers (Optional[int]): The maximum number of workers to use for streaming samples. If None, defaults to the system's default.

    Yields:
        Sample: A sample object corresponding to the given IDs.

    """
    infos = ei.experimental.data.get_sample_ids(
        filename=filename, category=category, labels=labels
    )
    ids = [info.sample_id for info in infos if info.sample_id is not None]
    return data.stream_samples_by_ids(ids, max_workers=max_workers)


def convert_sample_to_dataframe(
    sample, label_col_name: Optional[str] = "label", ts_col_name: Optional[str] = None
):
    """Converts a sample to a DataFrame and adds labels if provided.

    Args:
        sample (Sample): The sample to be converted.
        label_col_name (str): The name of the column for labels. Defaults to 'label'.
        ts_col_name (Optional[str]): The name of the column for timestamps. If None, timestamp information will be used as the index.

    Returns:
        pd.DataFrame: The converted DataFrame with optional labels and timestamps.

    Raises:
        Exception: If numpy or pandas is not installed.
    """
    if not (numpy_installed() and pandas_installed()):
        raise Exception(
            "Both Numpy and Pandas need to be installed in order to convert to a dataframe (pip install pandas numpy)"
        )

    data = json.loads(sample.data)
    df = _convert_cbor_to_df(data, ts_col_name=ts_col_name)
    if sample.structured_labels is not None:
        _add_labels_to_dataframe(df, sample.structured_labels, label_col_name)

    return df


def _add_labels_to_dataframe(df, labels, label_col_name="label"):
    df[label_col_name] = None
    for label_dict in labels:
        label = label_dict["label"]
        start_idx = label_dict["startIndex"]
        end_idx = label_dict["endIndex"]
        df.iloc[start_idx : end_idx + 1, df.columns.get_loc(label_col_name)] = label
    return df


def _convert_cbor_to_df(data, ts_col_name=None):
    import pandas as pd
    import numpy as np

    interval_ms = data["payload"]["interval_ms"]
    sensors = [sensor["name"] for sensor in data["payload"]["sensors"]]
    values = data["payload"]["values"]

    df = pd.DataFrame(values, columns=sensors)
    indices = np.arange(len(data["payload"]["values"])) * interval_ms
    if ts_col_name is not None:
        df[ts_col_name] = indices
    else:
        df.index = indices
    return df


__all__ = [
    "convert_sample_to_dataframe",
    "fetch_samples",
]
