"""Use this module to download sample datasets from Edge Impulse CDN."""
import os
import requests
import zipfile
import tarfile
import logging
from typing import List

DATASETS = [
    {
        "url": "https://cdn.edgeimpulse.com/datasets/activity2.zip",
        "name": "activity2",
        "description": "Dataset for activity recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cars-10.zip",
        "name": "cars-10",
        "description": "Dataset containing images of 10 different car models.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cars-2.zip",
        "name": "cars-2",
        "description": "Dataset containing images of 2 different car models.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cars-3.zip",
        "name": "cars-3",
        "description": "Dataset containing images of 3 different car models.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/coffee-lamp-bounding-boxes.zip",
        "name": "coffee-lamp-bounding-boxes",
        "description": "Dataset with bounding boxes for coffee and lamp images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/coffee_machine_stages.zip",
        "name": "coffee_machine_stages",
        "description": "Dataset for recognizing different stages of coffee making.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/coffee_machine_stages_multi_label.zip",
        "name": "coffee_machine_stages_multi_label",
        "description": "Dataset for recognizing multiple stages of coffee making.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/common-objects-image-recognition.zip",
        "name": "common-objects-image-recognition",
        "description": "Dataset for recognizing common objects in images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cough.zip",
        "name": "cough",
        "description": "Dataset for cough sound classification.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes-bounding-boxes.zip",
        "name": "cubes-bounding-boxes",
        "description": "Dataset with bounding boxes for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_coco-json-format.zip",
        "name": "cubes_coco-json-format",
        "description": "Dataset in COCO JSON format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_ei-object-detection-format.zip",
        "name": "cubes_ei-object-detection-format",
        "description": "Dataset in Edge Impulse object detection format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_open-image-csv-format.zip",
        "name": "cubes_open-image-csv-format",
        "description": "Dataset in Open Image CSV format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_pascal-voc-format.zip",
        "name": "cubes_pascal-voc-format",
        "description": "Dataset in Pascal VOC format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_plain-csv-od-format.zip",
        "name": "cubes_plain-csv-od-format",
        "description": "Dataset in plain CSV object detection format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/cubes_yolov5-format.zip",
        "name": "cubes_yolov5-format",
        "description": "Dataset in YOLOv5 format for cube images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/face-object-detection.zip",
        "name": "face-object-detection",
        "description": "Dataset for face detection.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/faucet.zip",
        "name": "faucet",
        "description": "Dataset for faucet detection.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/gestures-organization.zip",
        "name": "gestures-organization",
        "description": "Dataset for organizing gestures.",
    },
    # {
    #     "url": "https://cdn.edgeimpulse.com/datasets/gestures.parquet",
    #     "name": "gestures-parquet",
    #     "description": "Dataset for gestures in Parquet format.",
    # },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/gestures.zip",
        "name": "gestures",
        "description": "Dataset for gesture recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/heartrate.zip",
        "name": "heartrate",
        "description": "Dataset for heart rate monitoring.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/hello-world-keywords.zip",
        "name": "hello-world-keywords",
        "description": "Dataset for basic keyword recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/keywords.zip",
        "name": "keywords",
        "description": "Dataset for keyword recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/keywords2.zip",
        "name": "keywords2",
        "description": "Second dataset for keyword recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/keywords3.zip",
        "name": "keywords3",
        "description": "Third dataset for keyword recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/microbit-keywords-11khz.zip",
        "name": "microbit-keywords-11khz",
        "description": "Dataset for keyword recognition from microbit.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/object-detection-beer-vs-can.zip",
        "name": "object-detection-beer-vs-can",
        "description": "Dataset for detecting beer vs can.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/object-detection-streets.zip",
        "name": "object-detection-streets",
        "description": "Dataset for street object detection.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/plants-vs-lamps.zip",
        "name": "plants-vs-lamps",
        "description": "Dataset for distinguishing plants and lamps.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/regression.zip",
        "name": "regression",
        "description": "Dataset for regression analysis.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/visual-ad-plants-vs-lamps.zip",
        "name": "visual-ad-plants-vs-lamps",
        "description": "Visual advertisement dataset for plants vs lamps.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/visual-ad.zip",
        "name": "visual-ad",
        "description": "Visual advertisement dataset.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/visual-s.tar.gz",
        "name": "visual-s",
        "description": "Visual dataset (tar.gz).",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/visual-xs.tar.gz",
        "name": "visual-xs",
        "description": "Visual dataset (tar.gz).",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/visual.tar.gz",
        "name": "visual",
        "description": "Visual dataset (tar.gz).",
    },
]


def list_datasets() -> List[dict]:
    """List the available datasets on the Edge Impulse CDN."""
    return DATASETS


def download_dataset(name, force_redownload=False, overwrite_existing=False):
    """Download and extracts a dataset from the Edge Impulse CDN.

    Args:
        name (str): The name of the dataset to download.
        force_redownload (bool, optional): If True, forces re-downloading the dataset even if it exists. Defaults to False.
        overwrite_existing (bool, optional): If True, overwrites the existing dataset directory. Defaults to False.
    """
    ds = next(filter(lambda x: x["name"] == name, DATASETS), None)
    if ds is None:
        raise ValueError(
            f"Dataset {name} doesn't exist. Use `list_datasets()` to get a list of available datasets."
        )

    url = ds["url"]

    _download_dataset(
        url=url,
        extract_dir=f"datasets/{name}",
        overwrite_existing=overwrite_existing,
        force_redownload=force_redownload,
    )


def _download_dataset(
    url, extract_dir, force_redownload=False, overwrite_existing=False
):
    """Download and extract a dataset.

    Args:
        url (str): The URL of the dataset zip file.
        extract_dir (str): The directory to extract the dataset into.
        force_redownload (bool, optional): If True, forces re-downloading the dataset even if it exists. Defaults to False.
        overwrite_existing (bool, optional): If True, overwrites the existing dataset directory. Defaults to False.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/edgeimpulse/datasets")
    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists

    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_redownload:
        logger.info(f"Using cached file {cache_path}")
    else:
        logger.info(f"Downloading file from {url} to {cache_path}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            logger.error(
                f"Failed to download file from {url}. Status code: {response.status_code}"
            )
            raise Exception(
                f"Failed to download the dataset from {url}.  Status code: {response.status_code}"
            )

        total_size = int(response.headers.get("content-length", 0))

        chunk_size = 1024

        downloaded_size = 0
        with open(cache_path, "wb") as file:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                downloaded_size += len(data)
                progress = int(50 * downloaded_size / total_size)
                print(
                    "\r[{}{}] {:.2f}%".format(
                        "=" * progress,
                        " " * (50 - progress),
                        downloaded_size / total_size * 100,
                    ),
                    end="",
                    flush=True,
                )

        logger.info("Download complete")

    if cache_path.endswith(".zip"):
        _extract_zip_file(cache_path, extract_dir, overwrite_existing, logger)
    elif cache_path.endswith(".tar.gz"):
        _extract_tar_gz_file(cache_path, extract_dir, overwrite_existing, logger)
    else:
        logger.error(f"Unsupported file format for extraction: {cache_path}")


def _extract_zip_file(zip_file_path, extract_dir, overwrite_existing, logger):
    """Extract a zip file.

    Args:
        zip_file_path (str): The path to the zip file.
        extract_dir (str): The directory to extract the zip file into.
        overwrite_existing (bool): If True, overwrites the existing files. Defaults to False.
        logger (Logger): Logger instance for logging messages.
    """
    if not os.path.exists(extract_dir) or overwrite_existing:
        logger.info(f"Extracting zip file to {extract_dir}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extraction complete")
    else:
        logger.info(
            f"Extract directory '{extract_dir}' already exists. Skipping extraction."
        )


def _extract_tar_gz_file(tar_gz_file_path, extract_dir, overwrite_existing, logger):
    """Extract a tar.gz file.

    Args:
        tar_gz_file_path (str): The path to the tar.gz file.
        extract_dir (str): The directory to extract the tar.gz file into.
        overwrite_existing (bool): If True, overwrites the existing files. Defaults to False.
        logger (Logger): Logger instance for logging messages.
    """
    if not os.path.exists(extract_dir) or overwrite_existing:
        logger.info(f"Extracting tar.gz file to {extract_dir}")
        with tarfile.open(tar_gz_file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_dir)
        logger.info("Extraction complete")
    else:
        logger.info(
            f"Extract directory '{extract_dir}' already exists. Skipping extraction."
        )


# ruff: noqa: F821
def load_timeseries() -> "np.array":
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
