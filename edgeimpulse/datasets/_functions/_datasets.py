import logging
import os
import tarfile
from typing import List, Optional
import zipfile
import requests


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
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Audio+Classification+-+Keyword+Spotting.zip",
        "name": "keyword-spotting",
        "description": "Dataset for keyword spotting.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Sensor+Fusion+Classification+-+Coffee+machine+stages.zip",
        "name": "coffee-machine-stages-sensor-fusion",
        "description": "Sensor fusion dataset for coffee machine stages.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Cans+on+conveyor+belt.zip",
        "name": "cans-on-conveyor-belt",
        "description": "Object detection dataset for cans on a conveyor belt.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+Flat+washers.zip",
        "name": "flat-washers",
        "description": "Visual anomaly detection dataset for flat washers.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Audio+Classification+-+Faucet+vs+noise.zip",
        "name": "faucet-vs-noise",
        "description": "Dataset for classifying faucet sounds vs noise.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Motion+Classification+-+Continuous+motion+recognition.zip",
        "name": "continuous-motion-recognition",
        "description": "Dataset for continuous motion recognition.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Self+Attention+-+Cubes+on+conveyor+belt.zip",
        "name": "self-attention-cubes-on-conveyor-belt",
        "description": "Object detection dataset with self-attention for cubes on a conveyor belt.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Cubes+colors+on+conveyor+belt.zip",
        "name": "cubes-colors-on-conveyor-belt",
        "description": "Object detection dataset for colored cubes on a conveyor belt.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+Capsule.zip",
        "name": "capsule",
        "description": "Visual anomaly detection dataset for capsules.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Image+Classification+-+Microscope.zip",
        "name": "microscope",
        "description": "Image classification dataset for microscope images.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Vibration+Classification+-+Coffee+machine+stages.zip",
        "name": "coffee-machine-stages-vibration",
        "description": "Vibration classification dataset for coffee machine stages.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Audio+Classification+-+Glass+breaking.zip",
        "name": "glass-breaking",
        "description": "Audio classification dataset for glass breaking sounds.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Bottles+rack.zip",
        "name": "bottles-rack",
        "description": "Object detection dataset for bottles in a rack.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Dice.zip",
        "name": "dice",
        "description": "Object detection dataset for dice.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+DHT11.zip",
        "name": "dht11",
        "description": "Visual anomaly detection dataset for DHT11 sensors.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+Thermostatic+valves.zip",
        "name": "thermostatic-valves",
        "description": "Visual anomaly detection dataset for thermostatic valves.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Object+Detection+-+Dice+colors.zip",
        "name": "dice-colors",
        "description": "Object detection dataset for colored dice.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Image+Classification+-+Fire+extinguisher+safety+pin.zip",
        "name": "fire-extinguisher-safety-pin",
        "description": "Image classification dataset for fire extinguisher safety pins.",
    },
    {
        "url": "https://cdn.edgeimpulse.com/datasets/Visual+Anomaly+Detection+-+Fire+extinguisher+head+thread.zip",
        "name": "fire-extinguisher-head-thread",
        "description": "Visual anomaly detection dataset for fire extinguisher head threads.",
    },
]


def list_datasets() -> List[dict]:
    """List the available datasets on the Edge Impulse CDN."""
    return DATASETS


def download_dataset(
    name: str,
    force_redownload: bool = False,
    overwrite_existing: bool = False,
    show_progress: bool = False,
    extract_dir: Optional[str] = None,
):
    """Download and extract a dataset from the Edge Impulse CDN for tutorials, and quick prototyping.

    Saves the dataset in the `datasets/<name>` folder.
    Use `list_datasets` to show available datasets for download.

    Args:
        name (str): The name of the dataset to download.
        force_redownload (bool, optional): If True, forces the re-download of the dataset, even if it exists. Defaults to False.
        overwrite_existing (bool, optional): If True, overwrites the existing dataset directory. Defaults to False.
        show_progress (bool, optional): If True, outputs the download progress
    """
    ds = next(filter(lambda x: x["name"] == name, DATASETS), None)
    if ds is None:
        raise ValueError(
            f"Dataset {name} doesn't exist. Use `list_datasets()` to get a list of available datasets."
        )

    url = ds["url"]

    extract_dir = extract_dir or f"datasets/{name}"

    __download_dataset(
        url=url,
        extract_dir=extract_dir,
        overwrite_existing=overwrite_existing,
        force_redownload=force_redownload,
        show_progress=show_progress,
    )


def __download_dataset(
    url,
    extract_dir,
    force_redownload=False,
    overwrite_existing=False,
    show_progress=False,
):
    """Download and extract a dataset.

    Args:
        url (str): The URL of the dataset zip file.
        extract_dir (str): The directory to extract the dataset into.
        force_redownload (bool, optional): If True, forces re-downloading the dataset even if it exists. Defaults to False.
        overwrite_existing (bool, optional): If True, overwrites the existing dataset directory. Defaults to False.
        show_progress (bool, optional): If True, outputs the download progress
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/edgeimpulse/datasets")
    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists

    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_redownload:
        logging.info(f"Using cached file {cache_path}")
    else:
        logging.info(f"Downloading file from {url} to {cache_path}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            logging.error(
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
                if show_progress:
                    print(
                        "\r[{}{}] {:.2f}%".format(
                            "=" * progress,
                            " " * (50 - progress),
                            downloaded_size / total_size * 100,
                        ),
                        end="",
                        flush=True,
                    )

        logging.info("Download complete")

    if cache_path.endswith(".zip"):
        __extract_zip_file(cache_path, extract_dir, overwrite_existing)
    elif cache_path.endswith(".tar.gz"):
        __extract_tar_gz_file(cache_path, extract_dir, overwrite_existing)
    else:
        logging.error(f"Unsupported file format for extraction: {cache_path}")


def __extract_zip_file(zip_file_path, extract_dir, overwrite_existing):
    """Extract a zip file.

    Args:
        zip_file_path (str): The path to the zip file.
        extract_dir (str): The directory to extract the zip file into.
        overwrite_existing (bool): If True, overwrites the existing files. Defaults to False.
    """
    if not os.path.exists(extract_dir) or overwrite_existing:
        logging.info(f"Extracting zip file to {extract_dir}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info("Extraction complete")
    else:
        logging.info(
            f"Extract directory '{extract_dir}' already exists. Skipping extraction."
        )


def __extract_tar_gz_file(tar_gz_file_path, extract_dir, overwrite_existing):
    """Extract a tar.gz file.

    Args:
        tar_gz_file_path (str): The path to the tar.gz file.
        extract_dir (str): The directory to extract the tar.gz file into.
        overwrite_existing (bool): If True, overwrites the existing files. Defaults to False.
    """
    if not os.path.exists(extract_dir) or overwrite_existing:
        logging.info(f"Extracting tar.gz file to {extract_dir}")
        with tarfile.open(tar_gz_file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_dir)
        logging.info("Extraction complete")
    else:
        logging.info(
            f"Extract directory '{extract_dir}' already exists. Skipping extraction."
        )
