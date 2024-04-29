# noqa: D100
import edgeimpulse as ei
import json
import logging
import os
import pathlib

from edgeimpulse.data._functions.download import (
    download_samples_by_ids,
)
from edgeimpulse.data.sample_type import (
    Sample,
)

from edgeimpulse_api import (
    RawDataApi,
)

from edgeimpulse.util import (
    default_project_id_for,
)


def ids_from_succ(succ):  # noqa: D103
    return [sample.sample_id for sample in succ]


def assert_uploaded_samples(  # noqa: D103
    test, succ, check_label=False, check_structured_labels=False, check_meta=True
):
    ids = ids_from_succ(succ)
    samples = download_samples_by_ids(sample_ids=ids, show_progress=True)

    for upload in samples:
        found = [x.sample for x in succ if x.sample_id == upload.sample_id][0]

        test.assertEqual(found.sample_id, upload.sample_id)

        if check_meta:
            test.assertEqual(found.metadata, upload.metadata)

        if found.category != "split":
            test.assertEqual(found.category, upload.category)

        if upload.bounding_boxes:
            test.assertEqual(found.bounding_boxes, upload.bounding_boxes)

        if check_label:
            test.assertEqual(found.label, upload.label)

        if check_structured_labels:
            test.assertEqual(found.structured_labels, upload.structured_labels)


def delete_all_samples():  # noqa: D103
    """Delete all samples from the current logged in project. Use with care."""
    client = ei.util.configure_generic_client(
        key=ei.API_KEY,
        host=ei.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)
    raw_data_api = RawDataApi(client)
    raw_data_api.delete_all_samples(project_id=project_id)


# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)


# Build all datasets
def create_all_good_datasets():  # noqa: D103
    return [
        create_dataset_images(),
        create_dataset_good_csv(),
        create_dataset_wav(),
        create_dataset_video(),
        create_dataset_object_detection(),
        create_dataset_json(),
        create_dataset_cbor(),
    ]


# Build images dataset
def create_dataset_images():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "capacitor.01.png",
            "data": open(
                os.path.join(current_dir, dataset_dir, "capacitor.01.png"), "rb"
            ),
            "category": "training",
            "label": "capacitor",
            "metadata": {
                "source": "camera 1",
                "timestamp": "123",
            },
        },
        {
            "filename": "capacitor.02.png",
            "data": open(
                os.path.join(current_dir, dataset_dir, "capacitor.02.png"), "rb"
            ),
            "category": "training",
            "label": "capacitor",
            "metadata": {
                "source": "camera 2",
                "timestamp": "456",
            },
        },
    ]

    return [Sample(**i) for i in dataset]


# Build good CSV dataset
def create_dataset_good_csv():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "good.01.csv",
            "data": open(os.path.join(current_dir, dataset_dir, "good.01.csv"), "rb"),
            "category": "training",
            "label": "good",
            "metadata": {
                "source": "sensor 1",
                "timestamp": "123",
            },
        },
        {
            "filename": "good.02.txt",
            "data": open(os.path.join(current_dir, dataset_dir, "good.02.txt"), "rb"),
            "category": "training",
            "label": "good",
            "metadata": {
                "source": "sensor 2",
                "timestamp": "456",
            },
        },
    ]
    return [Sample(**i) for i in dataset]


# Build bad CSV dataset
def create_dataset_bad_csv():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "bad.01.csv",
            "data": open(os.path.join(current_dir, dataset_dir, "bad.01.csv"), "rb"),
            "category": "training",
            "label": "good",
            "metadata": {
                "source": "sensor 1",
                "timestamp": "123",
            },
        },
    ]
    return [Sample(**i) for i in dataset]


# Build wav dataset
def create_dataset_wav():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "hadouken.01.wav",
            "data": open(
                os.path.join(current_dir, dataset_dir, "hadouken.01.wav"), "rb"
            ),
            "category": "testing",
            "label": "hadouken",
            "metadata": {
                "source": "microphone",
                "timestamp": "123",
            },
        },
    ]
    return [Sample(**i) for i in dataset]


# Build video dataset
def create_dataset_video():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "moonwalk.01.avi",
            "data": open(
                os.path.join(current_dir, dataset_dir, "moonwalk.01.avi"), "rb"
            ),
            "category": "training",
            "label": "hadouken",
            "metadata": {
                "source": "camera",
                "timestamp": "123",
            },
        },
        {
            "filename": "moonwalk.02.mp4",
            "data": open(
                os.path.join(current_dir, dataset_dir, "moonwalk.02.mp4"), "rb"
            ),
            "category": "testing",
            "label": "hadouken",
            "metadata": {
                "source": "camera",
                "timestamp": "123",
            },
        },
    ]
    return [Sample(**i) for i in dataset]


# Build object detection dataset
def create_dataset_object_detection():  # noqa: D103
    # Set dataset dir
    dataset_dir = "sample_data/object_detection"
    current_dir = pathlib.Path(__file__).parent.resolve()

    # Construct object detection dataset and bounding box info for each sample
    dataset = []
    with open(
        os.path.join(current_dir, dataset_dir, "bounding_boxes.labels"), "r"
    ) as f:
        bb_info = json.load(f)
        for filename, bbs in bb_info["boundingBoxes"].items():
            dataset.append(
                {
                    "filename": filename,
                    "data": open(
                        os.path.join(current_dir, dataset_dir, filename), "rb"
                    ),
                    "category": "training",
                    "bounding_boxes": json.dumps(bbs),
                    "metadata": {
                        "source": "camera",
                        "timestamp": "123",
                    },
                }
            )

    return [Sample(**i) for i in dataset]


# Build JSON dataset
def create_dataset_json():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "wave.01.json",
            "data": open(os.path.join(current_dir, dataset_dir, "wave.01.json"), "rb"),
            "category": "training",
            "label": "wave",
            "metadata": {
                "source": "accelerometer",
                "timestamp": "123",
            },
        },
    ]
    return [Sample(**i) for i in dataset]


# Helper: build CBOR dataset
def create_dataset_cbor():  # noqa: D103
    dataset_dir = "sample_data"
    current_dir = pathlib.Path(__file__).parent.resolve()
    dataset = [
        {
            "filename": "wave.01.cbor",
            "data": open(os.path.join(current_dir, dataset_dir, "wave.01.cbor"), "rb"),
            "category": "training",
            "label": "wave",
            "metadata": {
                "source": "accelerometer",
                "timestamp": "123",
            },
        },
    ]
    return [Sample(**i) for i in dataset]
