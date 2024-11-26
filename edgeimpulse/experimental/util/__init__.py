"""This module contains all our experimental utilities.

Be warned these signatures may change.
"""

from edgeimpulse.util import numpy_installed, pandas_installed
from typing import Optional
import json

from edgeimpulse.experimental import data
import edgeimpulse as ei


def generate_labels_from_dataframe(df, label_col="label", file_name=None):
    """Generates structured labels from a DataFrame based on transitions in the specified label column.

    This function iterates over the rows of a pandas DataFrame and detects changes in the values of
    a specified label column. It groups consecutive rows with the same label value, and for each group,
    it returns a dictionary containing the start index, end index, and the label. Optionally, the result
    can be returned in a dictionary format, compatible with file saving, including the file name.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        label_col (str, optional): The column name that contains the labels. Defaults to 'label'.
        file_name (str, optional): If provided, the result is returned in a dictionary that can be used
                                   for saving structured labels to a file. Defaults to None.

    Returns:
        Union[list[dict],dict]: If `file_name` is not provided, a list of dictionaries is returned, where each
                           dictionary contains 'startIndex', 'endIndex', and 'label'. If `file_name` is provided,
                           a dictionary is returned in the format:
                           {
                               "version": 1,
                               "type": "structured-labels",
                               "structuredLabels": {
                                   file_name: [structured_labels]
                               }
                           }

    Raises:
        Exception: If either Numpy or Pandas is not installed.
    """
    if not (numpy_installed() and pandas_installed()):
        raise Exception(
            "Both Numpy and Pandas need to be installed in order to convert to a dataframe (pip install pandas numpy)"
        )

    structured_labels = []
    start_index = 0

    for i in range(1, len(df)):
        if df.iloc[i][label_col] != df.iloc[i - 1][label_col]:
            end_index = i - 1
            label = df.iloc[i - 1][label_col]
            structured_labels.append(
                {"startIndex": start_index, "endIndex": end_index, "label": label}
            )
            start_index = i

    structured_labels.append(
        {
            "startIndex": start_index,
            "endIndex": len(df) - 1,
            "label": df.iloc[-1][label_col],
        }
    )

    if not file_name:
        return structured_labels

    result = {
        "version": 1,
        "type": "structured-labels",
        "structuredLabels": {file_name: structured_labels},
    }
    return result


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
    df = convert_json_cbor_to_dataframe(data, ts_col_name=ts_col_name)
    if sample.structured_labels is not None:
        add_labels_to_dataframe(df, sample.structured_labels, label_col_name)

    return df


def add_labels_to_dataframe(df, labels, label_col_name="label"):
    """Adds labels to a DataFrame based on provided label information.

    Args:
        df (pandas.DataFrame): The DataFrame to which labels will be added.
        labels (list of dict): A list of dictionaries where each dictionary contains
            'label', 'startIndex', and 'endIndex' keys.
        label_col_name (str, optional): The name of the column where labels will be added.
            Defaults to "label".

    Returns:
        pandas.DataFrame: The DataFrame with the added labels.
    """
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of dictionaries.")

    df[label_col_name] = None

    for label_dict in labels:
        label = label_dict["label"]
        start_idx = label_dict["startIndex"]
        end_idx = label_dict["endIndex"]
        df.iloc[start_idx : end_idx + 1, df.columns.get_loc(label_col_name)] = label
    return df


def convert_json_cbor_to_dataframe(data, ts_col_name=None):
    """Converts JSON CBOR data to a pandas DataFrame.

    Args:
        data (dict): The JSON CBOR data containing payload information with 'interval_ms',
            'sensors', and 'values' keys.
        ts_col_name (str, optional): The name of the column to be used for time series data.
            If not provided, the DataFrame index will be used for time series data.

    Returns:
        pandas.DataFrame: A DataFrame with sensor values and optional time series data.
    """
    import pandas as pd
    import numpy as np

    if not isinstance(data, dict):
        raise ValueError("data must be a dict.")

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
    "convert_json_cbor_to_dataframe",
    "add_labels_to_dataframe",
    "generate_labels_from_dataframe",
]
