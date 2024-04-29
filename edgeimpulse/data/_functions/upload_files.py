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


def infer_category_and_label_from_filename(sample, file) -> None:
    """Extract label and category information from the filename and assigns them to the sample object.

    Files should look like this myfiles/training/wave.1.cbor where wave is label and training is the category.

    Args:
        sample (object): The sample object to which label and category will be assigned.
        file (str): The filename from which label and category information will be extracted.

    Returns:
        None
    """
    sample.label = os.path.basename(file).split(".")[0]
    if "testing" in file:
        sample.category = "testing"
    elif "training" in file:
        sample.category = "training"


def upload_directory(
    directory: str,
    category: str = None,
    label: str = None,
    metadata: dict = None,
    transform: Optional[callable] = None,
) -> UploadSamplesResponse:
    """Upload a directory of files to Edge Impulse.

    The files can be in CBOR, JSON, image, or WAV file formats. You can read more about the different file formats
    accepted by the Edge Impulse ingestion service here:

    https://docs.edgeimpulse.com/reference/ingestion-api

    Args:
        directory (str): The path to the directory containing the files to upload
        category (str): Category for the samples (train or split)
        label (str): Label for the files
        metadata (dict): Metadata to add to the file (visible in studio)
        transform (callable): A function to manipulate the sample and properties before uploading

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Examples:
        .. code-block:: python

            response = ei.experimental.data.upload_directory(directory="tests/sample_data/gestures")
            self.assertEqual(len(response.successes), 8)
            self.assertEqual(len(response.fails), 0)
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f"directory '{directory}' not found.")

    label_path = os.path.join(directory, LABEL_FILE)
    has_labels = os.path.exists(label_path)

    if has_labels:
        logging.debug("Label file found so using upload_dataset")
        return upload_exported_dataset(directory=directory, transform=transform)
    else:
        logging.debug("Label file not found so using upload_plain_directory")
        return upload_plain_directory(
            directory=directory,
            category=category,
            label=label,
            metadata=metadata,
            transform=transform,
        )


def upload_plain_directory(
    directory: str,
    category: str = None,
    label: str = None,
    metadata: dict = None,
    transform: Optional[callable] = None,
) -> UploadSamplesResponse:
    """Upload a directory of files to Edge Impulse.

    The samples can be in CBOR, JSON, image, or WAV file formats.

    Args:
        directory (str): The path to the directory containing the files to upload.
        category (str): Category for the samples
        label (str): Label for the files
        metadata (dict): Metadata to add to the file (visible in studio)
        transform (callable): A function to manipulate the sample and properties before uploading

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Examples:
        .. code-block:: python

            response = ei.experimental.data.upload_directory(directory="tests/sample_data/gestures")
            self.assertEqual(len(response.successes), 8)
            self.assertEqual(len(response.fails), 0)
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

    samples = []
    for file in files:
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

    res = upload_samples(samples)

    for sample in samples:
        sample.data.close()

    return res


def upload_exported_dataset(
    directory: str, transform: Optional[callable] = None
) -> UploadSamplesResponse:
    """Upload samples from a downloaded Edge Impulse dataset and preserving the `info.labels` information.

    Use this when you've exported your data in the studio.

    Args:
        directory (str): Path to the directory containing the dataset.
        transform (callable): A function to manipulate sample before uploading

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload.

    Raises:
        FileNotFoundError: If the labels file (info.labels) is not found in the specified directory.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f"directory '{directory}' not found.")

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

    samples = []

    for file, file_info in labels.items():
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

    res = upload_samples(samples)

    for sample in samples:
        sample.data.close()

    return res
