import unittest
import logging
import pathlib
import os
import warnings
import json

import edgeimpulse as ei
from edgeimpulse.data.sample_type import (
    Sample,
)

# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for uploading to complete
TIMEOUT = 1200.0  # 20 min


# Helper: delete sample from project
def delete_sample(filename, category=None):
    # Remove extension on the filename when querying the dataset in Studio
    filename_no_ext = os.path.splitext(filename)[0]

    # Get list of IDs that match the given sample filename
    ids = ei.data.get_ids_by_filename(
        filename=filename_no_ext,
        category=category,
        timeout_sec=TIMEOUT,
    )

    # Delete the IDs
    for id_num in ids:
        resp = ei.data.delete_sample_by_id(sample_id=id_num, timeout_sec=TIMEOUT)
        if resp is None:
            logging.warning(f"Could not delete sample {filename_no_ext}")


# Helper: build images dataset
def create_dataset_images():
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
    return dataset


# Helper: build good CSV dataset
def create_dataset_good_csv():
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
    return dataset


# Helper: build bad CSV dataset
def create_dataset_bad_csv():
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
    return dataset


# Helper: build wav dataset
def create_dataset_wav():
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
    return dataset


# Helper: build video dataset
def create_dataset_video():
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
    return dataset


# Helper: build object detection dataset
def create_dataset_object_detection():
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

    return dataset


# Helper: build JSON dataset
def create_dataset_json():
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
    return dataset


# Helper: build CBOR dataset
def create_dataset_cbor():
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
    return dataset


class TestDataUpload(unittest.TestCase):
    """
    Test upload features
    """

    def setUp(self):
        # Suppress annoying message from request's socket
        # https://github.com/psf/requests/issues/3912
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning
        )

    def test_incorrect_api_key(self):
        # Clobber config that's already been read from envvar
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"
        dataset = create_dataset_images()
        samples = (Sample(**i) for i in dataset)
        resps = ei.data.upload_samples(
            samples,
            allow_duplicates=False,
            timeout_sec=TIMEOUT,
        )
        ei.API_KEY = original_key

        # Check response
        resp_content = json.loads(resps[0].content.decode("utf-8"))
        self.assertEqual(resps[0].status_code, 401)
        self.assertFalse(resp_content["success"])
        self.assertEqual(resp_content["error"], "Invalid API key")

    def test_call_with_api_key(self):
        # Override incorrect API key with the correct key
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"

        # Generate samples
        dataset = create_dataset_images()
        samples = (Sample(**i) for i in dataset)

        # Try uploading samples
        try:
            # Upload samples
            resps = ei.data.upload_samples(
                samples,
                allow_duplicates=False,
                api_key=original_key,
                timeout_sec=TIMEOUT,
            )

            # Check responses
            for resp in resps:
                resp_content = json.loads(resp.content.decode("utf-8"))
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp_content["success"])
                for file in resp_content["files"]:
                    self.assertTrue(file["success"])

        # Raise any exceptions, always delete files from project
        except Exception as e:
            raise e
        finally:
            ei.API_KEY = original_key
            for sample in dataset:
                delete_sample(sample["filename"], sample["category"])

    def test_invalid_data(self):
        # Construct dataset with invalid data
        dataset = [
            {
                "filename": "wave.01.csv",
                "data": "1234567890",
                "category": "training",
                "label": "wave",
                "metadata": {
                    "source": "accelerometer 1",
                    "timestamp": "123",
                },
            },
        ]
        samples = (Sample(**i) for i in dataset)

        # Upload garbage data and check response
        try:
            resps = ei.data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )
            for resp in resps:
                resp_content = json.loads(resp.content.decode("utf-8"))
                self.assertEqual(resp.status_code, 500)
                self.assertFalse(resp_content["success"])

        # Raise any exceptions, always delete samples from project
        except Exception as e:
            raise e
        finally:
            for sample in dataset:
                delete_sample(sample["filename"], sample["category"])

    def test_upload_bad_csv(self):
        # Generate samples
        dataset = create_dataset_bad_csv()
        samples = (Sample(**i) for i in dataset)

        # Upload samples
        resps = ei.data.upload_samples(
            samples, allow_duplicates=False, timeout_sec=TIMEOUT
        )

        # Check responses
        try:
            for resp in resps:
                resp_content = json.loads(resp.content.decode("utf-8"))
                self.assertEqual(resp.status_code, 500)
                self.assertFalse(resp_content["success"])
                self.assertTrue(
                    str(resp_content["error"]).startswith(
                        "Could not parse this CSV file"
                    )
                )

        # Raise any exceptions
        except Exception as e:
            logging.error(f"Test failed. Response: {resp}")
            raise e

    def test_upload_files(self):
        # Generate samples
        datasets = [
            create_dataset_images(),
            create_dataset_good_csv(),
            create_dataset_wav(),
            create_dataset_video(),
            create_dataset_object_detection(),
            create_dataset_json(),
            create_dataset_cbor(),
        ]

        # Try uploading samples
        for dataset in datasets:
            # Check for files, upload, check again. Always delete files from project when done.
            try:
                # Make sure there are no files in the project that match the filename
                for sample_info in dataset:
                    filename = os.path.splitext(sample_info["filename"])[0]
                    ids = ei.data.get_ids_by_filename(
                        filename=filename,
                        category=sample_info["category"],
                        timeout_sec=TIMEOUT,
                    )
                    self.assertEqual(len(ids), 0)

                # Wrap the dataset info
                samples = (Sample(**i) for i in dataset)

                # Upload samples
                resps = ei.data.upload_samples(
                    samples, allow_duplicates=False, timeout_sec=TIMEOUT
                )

                # Check responses
                for resp in resps:
                    resp_content = json.loads(resp.content.decode("utf-8"))
                    self.assertEqual(resp.status_code, 200)
                    self.assertTrue(resp_content["success"])
                    for file in resp_content["files"]:
                        self.assertTrue(file["success"])

                # Verify that the files are in the project
                for sample_info in dataset:
                    filename = os.path.splitext(sample_info["filename"])[0]
                    ids = ei.data.get_ids_by_filename(
                        filename=filename,
                        timeout_sec=TIMEOUT,
                    )
                    self.assertEqual(len(ids), 1)

            # Raise any exceptions, always delete samples from project
            except Exception as e:
                raise e
            finally:
                resp = ei.data.delete_all_samples(
                    timeout_sec=TIMEOUT,
                )
                if resp is None:
                    logging.warning("Could not delete samples from project")

    def test_upload_duplicates(self):
        # Define dataset
        dataset = create_dataset_images()

        # Make sure there are no files in the project that match the filename
        try:
            for sample_info in dataset:
                filename = os.path.splitext(sample_info["filename"])[0]
                ids = ei.data.get_ids_by_filename(
                    filename=filename,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(ids), 0)
        except Exception as e:
            raise e
        finally:
            for sample in dataset:
                delete_sample(sample["filename"], sample["category"])

        # Try uploading the same samples twice with "allow_duplicates"
        for _ in range(2):
            # Generate samples
            samples = (Sample(**i) for i in dataset)

            # Check responses
            try:
                # Upload samples
                resps = ei.data.upload_samples(
                    samples, allow_duplicates=True, timeout_sec=TIMEOUT
                )

                for resp in resps:
                    resp_content = json.loads(resp.content.decode("utf-8"))
                    self.assertEqual(resp.status_code, 200)
                    self.assertTrue(resp_content["success"])
                    for file in resp_content["files"]:
                        self.assertTrue(file["success"])

            # Raise any exceptions and delete files from project
            except Exception as e:
                for sample in dataset:
                    delete_sample(sample["filename"], sample["category"])
                raise e

        # Ensure exactly 2 files with the same name have been uploaded
        try:
            for sample_info in dataset:
                filename = os.path.splitext(sample_info["filename"])[0]
                ids = ei.data.get_ids_by_filename(
                    filename=filename,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(ids), 2)
        except Exception as e:
            for sample in dataset:
                delete_sample(sample["filename"], sample["category"])
            raise e

        # Generate same samples
        samples = (Sample(**i) for i in dataset)

        # Upload samples without duplicates this time (should prevent uploading)
        resps = ei.data.upload_samples(
            samples, allow_duplicates=False, timeout_sec=TIMEOUT
        )

        # Check responses and always delete files from project
        try:
            for resp in resps:
                resp_content = json.loads(resp.content.decode("utf-8"))
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp_content["success"])
                for file in resp_content["files"]:
                    self.assertFalse(file["success"])
        except Exception as e:
            logging.error(f"Upload failed. Response: {resp}")
            raise e
        finally:
            for sample in dataset:
                delete_sample(sample["filename"], sample["category"])

    def test_get_filename_with_bad_id(self):
        # Try an ID that does not exist
        ret_filename = ei.data.get_filename_by_id(1)
        self.assertIsNone(ret_filename)

    def test_get_filename_with_good_id(self):
        # Define dataset
        dataset = create_dataset_images()

        # Check for files, upload, check again. Always delete files from project when done.
        try:
            # Make sure there are no files in the project that match the filename
            for sample_info in dataset:
                filename = os.path.splitext(sample_info["filename"])[0]
                ids = ei.data.get_ids_by_filename(
                    filename=filename,
                    category=sample_info["category"],
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(ids), 0)

            # Wrap the dataset info
            samples = (Sample(**i) for i in dataset)

            # Upload samples
            resps = ei.data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )

            # Check responses
            for resp in resps:
                resp_content = json.loads(resp.content.decode("utf-8"))
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp_content["success"])
                for file in resp_content["files"]:
                    self.assertTrue(file["success"])

            # Verify the returned filename is the same as the one given
            for sample_info in dataset:
                filename = os.path.splitext(sample_info["filename"])[0]
                ids = ei.data.get_ids_by_filename(
                    filename=filename,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(ids), 1)
                ret_filename = ei.data.get_filename_by_id(ids[0])
                self.assertEqual(ret_filename, filename)

        # Raise any exceptions, always delete samples from project
        except Exception as e:
            raise e
        finally:
            for category in ei.util.DATA_CATEGORIES:
                resp = ei.data.delete_all_samples(
                    category=category,
                    timeout_sec=TIMEOUT,
                )
                if resp is None:
                    logging.warning("Could not delete samples from project")
