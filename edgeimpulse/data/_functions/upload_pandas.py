# mypy: ignore-errors
# ruff: noqa: D100
# NOTE: We're not importing pandas here because we want to be compatible with pandas like
# frameworks. I.e. Dask, Modin, Polars etc.
# Please try to rely on duck typing where possible and don't make any dependencies towards
# specific frameworks
#
# In the end we'll adopt something like this https://data-apis.org/dataframe-protocol/latest/purpose_and_scope.html

import io
import random
import logging
import csv

from edgeimpulse.experimental import util

from edgeimpulse.data.sample_type import (
    Sample,
    UploadSamplesResponse,
)

from edgeimpulse.data._functions.upload import (
    upload_samples,
)
from typing import List, Optional, Literal


MSG_NO_DF_MODULE = (
    "DataFrame methods on input object not found. DataFrame support "
    "requires pandas (or similar) to be installed."
)


def row_metadata(row, metadata_col):
    """Get the metadata from a dataframe row for a single sample."""
    metadata = None
    if metadata_col is not None:
        metadata = {k: str(row[k]) for k in metadata_col}
    return metadata


def ts_to_ms(timestamp):
    """Convert timestamps to milliseconds for the EI data acquisition format."""
    if isinstance(timestamp, (int, float)):
        return int(timestamp * 1000)
    elif hasattr(timestamp, "timestamp"):
        return int(timestamp.timestamp() * 1000)
    else:
        raise Exception(f"Error: Unsupported timestamp format - {type(timestamp)}")


def upload_pandas_sample(
    df,
    label: Optional[str] = None,
    sample_rate_ms: Optional[int] = None,
    filename: Optional[str] = None,
    axis_columns: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
    label_col: Optional[str] = None,
    category: Literal["training", "testing", "split"] = "split",
) -> UploadSamplesResponse:
    """Upload a single dataframe sample.

    Upload a single dataframe sample to Edge Impulse.

    Args:
        df (DataFrame): The input DataFrame containing data.
        label (str, optional): The label for the sample. Default is None.
        sample_rate_ms (int, optional): The sampling rate of the time series data (in milliseconds).
        filename (str, optional): The filename for the sample. Default is None.
        axis_columns (List[str], optional): List of column names representing axis if the data is
            multi-dimensional. Default is None.
        metadata (dict, optional): Dictionary containing metadata information. Default is None.
        label_col (str, optional): When given, this is used for the multi-label
        category (str or None, optional): Category or class label for the entire dataset. Default is split.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        AttributeError: If the input object does not have a `reset_index` method.
        ValueError: If the `axis_columns` argument is not a list of strings or if the `metadata`
            argument is not a dictionary.

    Examples:
        Uploads a pandas dataframe as single sample

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data
        import pandas as pd

        # Construct one dataframe for each sample (multidimensional, non-time series)
        df = pd.DataFrame([[-9.81, 0.03, 0.21]], columns=["accX", "accY", "accZ"])

        # Optional metadata for all samples being uploaded
        metadata = {
            "source": "accelerometer",
            "collection site": "desk",
        }

        # Upload the sample
        response = data.upload_pandas_sample(
            df,
            label="One",
            filename="001",
            metadata=metadata,
            category="training",
        )
        ```
    """
    sample = pandas_dataframe_to_sample(
        df,
        label=label,
        sample_rate_ms=sample_rate_ms,
        filename=filename,
        axis_columns=axis_columns,
        label_col=label_col,
        metadata=metadata,
        category=category,
    )
    return upload_samples(sample)


def pandas_dataframe_to_sample(
    df,
    sample_rate_ms: Optional[int] = None,
    label: Optional[str] = None,
    filename: Optional[str] = None,
    axis_columns: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
    label_col: Optional[str] = None,
    category: Literal["training", "testing", "split"] = "split",
) -> Sample:
    """Convert a dataframe to a single sample. Can handle both *timeseries* and *non-timeseries* data.

    In order to be inferred as timeseries it must have:

    - More than one row
    - A sample rate or an index from which the sample rate can be inferred
        - Therefore must be monotonically increasing
        - And int or a date

    Args:
        df (DataFrame): The input DataFrame containing data.
        sample_rate_ms (int): The sampling rate of the time series data (in milliseconds).
        label (str, optional): The label for the sample. Default is None.
        filename (str, optional): The filename for the sample. Default is None.
        axis_columns (List[str], optional): List of column names representing axis if the data is multi-dimensional. Default is None.
        metadata (dict, optional): Dictionary containing metadata information for the sample. Default is None.
        label_col (str, optional): When timeseries and multilabel, specify the column here to mark the dataset as multilabel
        category (str or None, optional): To which category this sample belongs (training/testing/split). Default is split.

    Returns:
        Sample: A sample object containing the data from the dataframe.
    """
    # Check to make sure dataframe operations are supported
    if not hasattr(df, "reset_index") or not callable(df.reset_index):
        raise AttributeError(MSG_NO_DF_MODULE)

    if axis_columns is not None:
        if not isinstance(axis_columns, list) or not all(
            isinstance(col, str) for col in axis_columns
        ):
            raise ValueError("The 'axis_columns' argument must be a list of strings.")

    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("The 'metadata' argument must be a dictionary or None.")

    if axis_columns:
        df = df[axis_columns]

    # if more than one row is given, it must be a timeseries (this is what the
    # ingestion csv format currently requires)
    is_time_series = len(df) > 1

    # if we have more than one row, we can assume we're with timeseries according
    # to the csv ingestion docs https://docs.edgeimpulse.com/reference/importing-csv-data
    structured_labels = None

    if is_time_series:
        if label_col:
            structured_labels = util.generate_labels_from_dataframe(
                df, label_col=label_col
            )
            df = df.drop(columns=[label_col])

        if sample_rate_ms:
            # we build our own index column based on the sample rate since csv
            # upload needs a column named timestamp thats increasing monotonically when
            # dealing with timeseries
            df.reset_index(drop=True, inplace=True)
            df.index = df.index * int(sample_rate_ms)
        else:
            logging.debug("Trying to infer sample rate")

            if not hasattr(df.index, "is_monotonic_increasing"):
                raise ValueError(
                    "Index should be monotonically increasing in order to detect sample rate. Or specify sample_rate_ms argument."
                )

            if hasattr(df.index, "seconds"):
                # the csv ingestion services requires the timestamp to be specified
                # in milliseconds. See here: https://docs.edgeimpulse.com/reference/importing-csv-data
                # So here we convert the index when its timedelta index.
                # Because we're duck typing we can't check for
                # explicit TimeDeltaIndex
                df.index = df.index.seconds * 1000
            else:
                # Must be a normal range based index, so we can use that.
                pass

    # We convert here to a csv since it's a little bit easier to use than the JSON ei data
    # We don't need sensors here (they are automatically inferred)
    # And we can both support timeseries and non-time series.

    csv = io.StringIO()
    df.to_csv(csv, index=is_time_series, index_label="timestamp")

    # Print the first 2000 characters of the csv to the log
    max_chars = 2000
    csv.seek(0, 2)
    num_char = csv.tell()
    csv.seek(0)
    logging.debug(
        "Csv file to be uploaded:\r\n------\r\n%s%s------",
        csv.read(max_chars),
        " ...\r\n" if num_char > max_chars else "",
    )
    csv.seek(0)

    if not filename:
        filename = "%08x" % random.getrandbits(64)

    sample = Sample(
        filename=f"{filename}.csv",
        metadata=metadata,
        label=label,
        data=csv,
        structured_labels=structured_labels,
        category=category,
    )

    return sample


def upload_pandas_dataframe_wide(
    df,
    sample_rate_ms: int,
    data_col_start: Optional[int] = None,
    label_col: Optional[str] = None,
    category_col: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
    data_col_length: Optional[int] = None,
    data_axis_cols: Optional[List[str]] = None,
) -> UploadSamplesResponse:
    """Upload a dataframe to Edge Impulse where each column represents a value in the timeseries data and the rows become the individual samples.

    Args:
        df (DataFrame): The input DataFrame containing time series data.
        data_col_start (int): The index of the column from which the time series data begins.
        sample_rate_ms (int): The sampling rate of the time series data (in milliseconds).
        label_col (str, optional): The column name containing labels for each
            time series. Default is None.
        category_col (str, optional): The column name containing the category for the
            data. Default is None.
        metadata_cols (List[str], optional): List of column names containing metadata
            information. Default is None.
        data_col_length (int, optional): The number of columns that represent a single
            time series. Default is None.
        data_axis_cols (List[str], optional): List of column names representing the axis if the data is
            multidimensional. Default is None.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        AttributeError: If the input object does not have a `iterrows` method.
        ValueError: If the `data_col_length` argument is not an integer or if the `data_col_start`
            argument is not an integer.

    Examples:
        Uploads a panda dataframe

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data

        import pandas as pd

        values = [
            [1, "idle", 0.8, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.8],  # ...continued
            [2, "motion", 0.3, 0.9, 0.4, 0.6, 0.8, 0.9, 0.5, 0.4],  # ...continued
        ]

        df = pd.DataFrame(
            values, columns=["id", "label", "0", "1", "2", "3", "4", "5", "6", "7"]
        )

        response = data.upload_pandas_dataframe_wide(
            df,
            label_col="label",
            metadata_col=["id"],
            data_col_start=2,
            sample_rate_ms=100,
        )
        assert(len(response.successes)==2)
        assert(len(response.fails)==0)
        ```
    """
    # Check to make sure dataframe operations are supported
    if not hasattr(df, "iterrows") or not callable(df.iterrows):
        raise AttributeError(MSG_NO_DF_MODULE)

    samples = []
    is_single_axis = data_axis_cols is None
    if is_single_axis:
        for _, row in df.iterrows():
            # We need to transpose the single wide row to a timeseries long dataframe
            # We reset the index in order to make it compatible. After to frame, the column names are the index
            # so if that is text it will break the upload

            row_df = row[data_col_start:].to_frame()  # transforms from wide to long.
            row_df.reset_index(
                drop=True, inplace=True
            )  # the index becomes to column names so lets drop them and convert to a standard range

            sample = pandas_dataframe_to_sample(row_df, sample_rate_ms=sample_rate_ms)
            sample.label = row.get(label_col, None)
            sample.category = row.get(category_col, None)
            sample.metadata = row_metadata(row, metadata_cols)

            samples.append(sample)
    else:
        # must be multi axis
        if data_col_length is None:
            raise ValueError("The data_col_length must be set to an integer")

        dim_columns = [
            [f"{col}{i}" for col in data_axis_cols] for i in range(data_col_length)
        ]

        all_columns = [col for sub in dim_columns for col in sub]
        missing_columns = [col for col in all_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Can't find the following columns: {missing_columns}")

        csv_cols = ["timestamp", *data_axis_cols]

        for index, row in df.iterrows():
            data = []

            for index, res in enumerate(dim_columns):
                timestamp = index * sample_rate_ms
                row_values = row[res].values.tolist()
                data.append([timestamp, *row_values])

            # This CSV logic is duplicate but since we're duck typing
            # we can't make a new dataframe.  And because of that we need to jump trough hoops
            # to serialize this to csv.
            csv_data = io.StringIO()
            writer = csv.writer(csv_data)
            writer.writerow(csv_cols)
            writer.writerows(data)
            csv_data.seek(0)

            logging.info(
                "Csv file to be uploaded:\n------\n%s------", csv_data.getvalue()
            )

            filename = "%08x" % random.getrandbits(64)

            sample = Sample(filename=f"{filename}.csv", data=csv_data)
            sample.label = row.get(label_col, None)
            sample.category = row.get(category_col, None)
            sample.metadata = row_metadata(row, metadata_cols)

            samples.append(sample)

    return upload_samples(samples)


def upload_pandas_dataframe(
    df,
    feature_cols: List[str],
    label_col: Optional[str] = None,
    category_col: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
) -> UploadSamplesResponse:
    """Upload non-timeseries data to Edge Impulse where each dataframe row becomes a sample.

    Args:
        df (dataframe): The DataFrame to be uploaded.
        feature_cols (List[str]): A list of column names containing features.
        label_col (str, optional): The name of the column containing labels for the data.
        category_col (str, optional): The name of the column containing the category for the data.
        metadata_cols (List[str], optional): Optional list of column names containing metadata.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        AttributeError: If the input object does not have an `iterrows` method.

    Examples:
        Uploads a Pandas dataframe

        ```python
        import pandas as pd

        from edgeimpulse import data

        # Construct non-time series data, where each row is a different sample
        sample_data = [
            ["desk", "training", "One", -9.81, 0.03, 0.21],
            ["field", "training", "Two", -9.56, 5.34, 1.21],
        ]
        columns = ["location", "category", "label", "accX", "accY", "accZ"]

        # Wrap the data in a DataFrame
        df = pd.DataFrame(sample_data, columns=columns)

        # Upload non-time series DataFrame (with multiple samples) to the project
        response = data.upload_pandas_dataframe(
            df,
            feature_cols=["accX", "accY", "accZ"],
            label_col="label",
            category_col="category",
            metadata_cols=["location"],
        )

        assert len(response.fails) == 0, "Could not upload some files"
        ```
    """
    # Check to make sure dataframe operations are supported
    if not hasattr(df, "iterrows") or not callable(df.iterrows):
        raise AttributeError(MSG_NO_DF_MODULE)

    samples = []
    for index, row in df.iterrows():
        row_df = row.to_frame().transpose()[feature_cols]
        sample = pandas_dataframe_to_sample(row_df)

        sample.label = row.get(label_col, None)
        sample.category = row.get(category_col, None)
        sample.metadata = row_metadata(row, metadata_cols)

        samples.append(sample)

    return upload_samples(samples)


def upload_pandas_dataframe_with_group(
    df,
    timestamp_col: str,
    group_by: str,
    feature_cols: List[str],
    label_col: Optional[str] = None,
    category_col: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
) -> UploadSamplesResponse:
    """Upload a dataframe where the rows contain multiple samples and timeseries data for those samples.

    It uses a `group_by` in order to detect what timeseries value belongs
    to which sample.

    Args:
        df (dataframe):
            The DataFrame to be uploaded.
        timestamp_col (str):
            The name of the column containing the timestamp for the data (in seconds).
        group_by (str):
            The name of the column containing the group for the data.
        feature_cols (List[str]):
            A list of column names containing features.
        label_col (str, optional): The name of the column containing labels for the data. Each group
            must have the same label. Default is None (derived from group name).
        category_col (str, optional): The name of the column containing the category for the data.
            Each group must have the same category. Default is None (random training/test split).
        metadata_cols (List[str], optional): A list of column names containing metadata information.
            Each group must have the same metadata. Default is None.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Examples:
        Uploads a dataframe

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data
        import pandas as pd

        sample_data = [
            ["desk", "sample 1", "training", "idle", 0, -9.81, 0.03, 0.21],
            ["desk", "sample 1", "training", "idle", 0.01, -9.83, 0.04, 0.27],
            ["desk", "sample 1", "training", "idle", 0.02, -9.12, 0.03, 0.23],
            ["desk", "sample 1", "training", "idle", 0.03, -9.14, 0.01, 0.25],
            ["field", "sample 2", "training", "wave", 0, -9.56, 5.34, 1.21],
            ["field", "sample 2", "training", "wave", 0.01, -9.43, 1.37, 1.27],
            ["field", "sample 2", "training", "wave", 0.02, -9.22, -4.03, 1.23],
            ["field", "sample 2", "training", "wave", 0.03, -9.50, -0.98, 1.25],
        ]

        columns = ["location", "sample_name", "category", "label", "timestamp", "accX", "accY", "accZ"]
        df = pd.DataFrame(sample_data, columns=columns)

        # Upload time series DataFrame (with multiple samples and multiple dimensions) to the project
        response = data.upload_pandas_dataframe_with_group(
            df,
            group_by="sample_name",
            timestamp_col="timestamp",
            feature_cols=["accX", "accY", "accZ"],
            label_col="label",
            category_col="category",
            metadata_cols=["location"]
        )
        assert len(response.fails) == 0, "Could not upload some files"
        ```
    """
    # Check to make sure dataframe operations are supported
    if not hasattr(df[timestamp_col], "apply") or not callable(df[timestamp_col].apply):
        raise AttributeError(MSG_NO_DF_MODULE)

    # Convert timestamps to milliseconds
    samples = []
    df[timestamp_col] = df[timestamp_col].apply(ts_to_ms)

    # If the timestamp column is not labeled "timestamp", rename it
    hard_timestamp_col = "timestamp"
    if timestamp_col != hard_timestamp_col:
        # Append 4 random hex digits to columns named "timestamp" to avoid conflicts
        while hard_timestamp_col in df.columns:
            col_name = f"{hard_timestamp_col}_{random.getrandbits(16):04x}"
            df.rename(columns={hard_timestamp_col: col_name}, inplace=True)

        # Rename the timestamp column
        df.rename(columns={timestamp_col: hard_timestamp_col}, inplace=True)

    # Get unique groups
    groups = df[group_by].unique()

    # Iterate over groups
    for group in groups:
        group_df = df[df[group_by] == group][[hard_timestamp_col, *feature_cols]]
        group_df = group_df.sort_values(by=hard_timestamp_col)

        # Extract category
        category = None
        if category_col:
            categories = list(df[df[group_by] == group][category_col].unique())
            if len(categories) > 1:
                raise ValueError(f'More than one category found for group "{group}"')
            category = categories[0]

        # Extract label
        label = None
        if label_col:
            labels = list(df[df[group_by] == group][label_col].unique())
            if len(labels) > 1:
                raise ValueError(f'More than one label found for group "{group}"')
            label = labels[0]

        # Extract metadata
        metadata = None
        if metadata_cols:
            metadata = {}
            for metadata_col in metadata_cols:
                metadatas = list(df[df[group_by] == group][metadata_col].unique())
                if len(metadatas) > 1:
                    raise ValueError(
                        "More than one metadata value found for "
                        f'{metadata_col}" in group "{group}"'
                    )
                metadata[metadata_col] = metadatas[0]

        csv = io.StringIO()
        group_df.to_csv(csv, index=False)
        csv.seek(0)

        sample = Sample(
            filename=f"{group}.csv",
            data=csv,
            category=category,
            label=label,
            metadata=metadata,
        )
        samples.append(sample)

    return upload_samples(samples)
