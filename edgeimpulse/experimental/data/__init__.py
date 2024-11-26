"""Use this module to upload data to Edge Impulse."""

from edgeimpulse.data._functions.delete import (
    delete_all_samples,
    delete_sample_by_id,
    delete_samples_by_filename,
)

from edgeimpulse.data._functions.download import (
    download_samples_by_ids,
    stream_samples_by_ids,
)

from edgeimpulse.data._functions.upload_files import (
    upload_plain_directory,
    upload_exported_dataset,
    upload_directory,
)

from edgeimpulse.data._functions.upload_files import (
    infer_from_filename as infer_category_and_label_from_filename,
)


from edgeimpulse.data._functions.upload import upload_samples

from edgeimpulse.data._functions.upload_pandas import (
    upload_pandas_dataframe,
    upload_pandas_dataframe_wide,
    upload_pandas_sample,
    upload_pandas_dataframe_with_group,
    pandas_dataframe_to_sample,
)

from edgeimpulse.data._functions.util import (
    get_filename_by_id,
    get_sample_ids,
)

from edgeimpulse.data._functions.upload_numpy import (
    upload_numpy,
    numpy_timeseries_to_sample,
)

from edgeimpulse.data.sample_type import Sample

__all__ = [
    "delete_all_samples",
    "get_sample_ids",
    "delete_sample_by_id",
    "delete_samples_by_filename",
    "get_filename_by_id",
    "upload_samples",
    "upload_exported_dataset",
    "upload_directory",
    "upload_plain_directory",
    "upload_pandas_dataframe",
    "upload_pandas_dataframe_wide",
    "upload_pandas_sample",
    "upload_pandas_dataframe_with_group",
    "pandas_dataframe_to_sample",
    "upload_numpy",
    "numpy_timeseries_to_sample",
    "download_samples_by_ids",
    "stream_samples_by_ids",
    "infer_category_and_label_from_filename",
    "Sample",
]
