# mypy: ignore-errors
# ruff: noqa: D100
from edgeimpulse.data.sample_type import Sample
from typing import Optional
from glob import glob
import os
import json
from fnmatch import fnmatch
import logging

from edgeimpulse.data._functions.upload import (
    upload_samples,
)
from edgeimpulse.data.sample_type import (
    UploadSamplesResponse,
)

LABEL_FILE = "info.labels"
ALLOWED_FILES = [
    "*.jpg",
    "*.png",
    "*.mp4",
    "*.avi",
    "*.wav",
    "*.cbor",
    "*.json",
    "*.csv",
    "info.labels",
]


def infer_from_filename(sample: Sample, file: str) -> None:
    """Extract label and category information from the filename and assigns them to the sample object.

    Files should look like this `my-dataset/training/wave.1.cbor` where `wave` is the label and `training` is the category.
    It checks if there is `training`, `testing` or `anomaly` in the filename to determine the sample category.

    Args:
        sample (Sample): The sample object to which the label and category will be assigned.
        file (str): The filename from which label and category information will be extracted.

    Returns:
        None
    """
    sample.label = os.path.basename(file).split(".")[0]

    if "testing" in file:
        sample.category = "testing"
    elif "anomaly" in file:
        sample.category = "anomaly"
    elif "training" in file:
        sample.category = "training"


def upload_directory(
    directory: str,
    category: Optional[str] = None,
    label: Optional[str] = None,
    metadata: Optional[dict] = None,
    transform: Optional[callable] = None,
    allow_duplicates: Optional[bool] = False,
    show_progress: Optional[bool] = False,
    batch_size: Optional[int] = 1024,
) -> UploadSamplesResponse:
    """Upload a directory of files to Edge Impulse.

    Tries to autodetect whether it's an Edge Impulse exported dataset, or a standard directory. The files can be in CBOR, JSON, image, or
    WAV file formats. You can read more about the different file formats accepted by the Edge Impulse ingestion service here:

    https://docs.edgeimpulse.com/reference/ingestion-api

    Args:
        directory (str): The path to the directory containing the files to upload
        category (str): Category for the samples "training", "testing", "anomaly", "split"
        label (str): Label for the samples
        metadata (dict): Metadata to add to the samples (visible in studio)
        transform (callable): A function to manipulate the sample and properties before uploading
        allow_duplicates (Optional[bool]): Set to `True` to allow samples with the same data to be
            uploaded. If `False`, the ingestion service will perform a hash of the data and compare
            it to the hashes of the data already in the project. If a match is found, the service
            will reject the incoming sample (uploading for other samples will continue).
        show_progress (Optional[bool]): Show progress bar while uploading samples. Default is `False`.
        batch_size (Optional[int]): The number of samples to upload in a single batch. Default is 1024.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Examples:
        Upload a directory

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data
        response = data.upload_directory(directory="tests/sample_data/gestures")

        print(len(response.successes) == 8)
        print(len(response.fails) == 0)
        ```
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f"directory '{directory}' not found.")

    label_path = os.path.join(directory, LABEL_FILE)
    has_labels = os.path.exists(label_path)

    if has_labels:
        logging.debug(
            "Edge Impulse label file found so using `upload_exported_dataset`"
        )
        return upload_exported_dataset(
            directory=directory,
            transform=transform,
            allow_duplicates=allow_duplicates,
            show_progress=show_progress,
            batch_size=batch_size,
        )
    else:
        logging.debug(
            "No Edge Impulse label file found so using `upload_plain_directory`"
        )
        return upload_plain_directory(
            directory=directory,
            category=category,
            label=label,
            metadata=metadata,
            transform=transform,
            allow_duplicates=allow_duplicates,
            show_progress=show_progress,
            batch_size=batch_size,
        )


def upload_plain_directory(
    directory: str,
    category: Optional[str] = None,
    label: Optional[str] = None,
    metadata: Optional[dict] = None,
    transform: Optional[callable] = None,
    allow_duplicates: Optional[bool] = False,
    show_progress: Optional[bool] = False,
    batch_size: Optional[int] = 1024,
) -> UploadSamplesResponse:
    """Upload a directory of files to Edge Impulse.

    The samples can be in CBOR, JSON, image, or WAV file formats.

    Args:
        directory (str): The path to the directory containing the files to upload.
        category (str): The category for the samples "training", "testing", "anomaly", "split".
        label (str): The label for the samples.
        metadata (dict): Metadata to add to the samples (visible in studio).
        transform (callable): A function to manipulate the sample and properties before uploading.
        allow_duplicates (Optional[bool]): Set to `True` to allow samples with the same data to be
            uploaded. If `False`, the ingestion service will perform a hash of the data and compare
            it to the hashes of the data already in the project. If a match is found, the service
            will reject the incoming sample (uploading for other samples will continue).
        show_progress (Optional[bool]): Show the progress bar while uploading samples. Default is `False`.
        batch_size (Optional[int]): The number of samples to upload in a single batch. The default is 1024.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Examples:
        Uploads a plain directory

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data

        response = data.upload_directory(directory="tests/sample_data/gestures")
        assert(len(response.successes )== 8)
        assert(len(response.fails) == 0)
        ```
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f"directory '{directory}' not found.")

    files = glob(os.path.join(directory, "**", "*"), recursive=True)
    files = [file for file in files if os.path.isfile(file)]
    files = [
        file
        for file in files
        if any(fnmatch(os.path.basename(file), pattern) for pattern in ALLOWED_FILES)
    ]

    # Upload samples in batches
    results = None
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        samples = []
        for file in batch:
            sample = Sample(
                data=open(file, "rb"),
                filename=os.path.basename(file),
                metadata=metadata,
                category=category,
                label=label,
            )

            if transform:
                transform(sample, file)

            samples.append(sample)

        res = upload_samples(
            samples, allow_duplicates=allow_duplicates, show_progress=show_progress
        )

        # Merge results
        if results is None:
            results = res
        else:
            results.extend(res.successes, res.fails)

        # Close file handles
        for sample in samples:
            sample.data.close()

    return results


def upload_exported_dataset(
    directory: str,
    transform: Optional[callable] = None,
    allow_duplicates: Optional[bool] = False,
    show_progress: Optional[bool] = False,
    batch_size: Optional[int] = 1024,
) -> UploadSamplesResponse:
    """Upload samples from a downloaded Edge Impulse dataset and preserve the `info.labels` information.

    Use this when you've exported your data in the studio, via the `export` functionality.

    Args:
        directory (str): Path to the directory containing the dataset.
        transform (callable): A function to manipulate sample before uploading.
        allow_duplicates (Optional[bool]): Set to `True` to allow samples with the same data to be
            uploaded. If `False`, the ingestion service will perform a hash of the data and compare
            it to the hashes of the data already in the project. If a match is found, the service
            will reject the incoming sample (uploading for other samples will continue).
        show_progress (Optional[bool]): Show progress bar while uploading samples. Default is `False`.
        batch_size (Optional[int]): The number of samples to upload in a single batch. Default is 1024.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the labels file (info.labels) is not found in the specified directory.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    label_path = os.path.join(directory, LABEL_FILE)
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Labels file '{LABEL_FILE}' not found in the specified directory."
        )

    labels = {}
    with open(label_path) as file:
        labels = json.load(file)
        labels = {
            os.path.join(directory, file["path"]): file for file in labels["files"]
        }

    # Upload samples in batches
    results = None
    for i in range(0, len(labels), batch_size):
        batch = list(labels.items())[i : i + batch_size]
        samples = []
        for file, file_info in batch:
            sample = Sample(
                data=open(file, "rb"),
                bounding_boxes=file_info["boundingBoxes"],
                filename=os.path.basename(file),
                structured_labels=file_info["label"].get("labels", None),
                metadata=file_info.get("metadata", None),
                category=file_info.get("category", None),
                label=file_info["label"].get("label", None),
            )

            if transform:
                transform(sample, file)

            samples.append(sample)

        res = upload_samples(
            samples, allow_duplicates=allow_duplicates, show_progress=show_progress
        )

        # Merge results
        if results is None:
            results = res
        else:
            results.extend(res.successes, res.fails)

        # Close file handles
        for sample in samples:
            sample.data.close()

    return results
