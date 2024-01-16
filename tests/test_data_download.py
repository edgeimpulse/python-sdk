import unittest
import logging
import os
import random
import warnings

import edgeimpulse as ei
from edgeimpulse.data._functions.delete import (
    delete_all_samples,
)
from edgeimpulse.data._functions.download import (
    download_samples_by_ids,
)
from edgeimpulse.data._functions.upload import (
    upload_samples,
)
from edgeimpulse.data._functions.util import (
    get_sample_ids,
)
from edgeimpulse_api.exceptions import (
    UnauthorizedException,
)

from . import util


# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for uploading to complete
TIMEOUT = 1200.0  # 20 min


# Helper function: upload dataset
def upload_dataset(samples):
    # Make sure there are no files in the project that match the filename
    for sample in samples:
        filename = os.path.splitext(sample.filename)[0]
        infos = get_sample_ids(
            filename=filename,
            category=sample.category,
            timeout_sec=TIMEOUT,
        )
        if len(infos) > 0:
            raise RuntimeError(
                f"Found {len(infos)} samples with filename {filename} in category "
                f"{sample.category}. Please delete these samples before running this "
                "test."
            )

    # Upload samples and get IDs
    _ = upload_samples(samples, allow_duplicates=False, timeout_sec=TIMEOUT)
    ids = [sample.sample_id for sample in samples]

    return ids


class TestDataDownload(unittest.TestCase):
    """
    Test downloading data from Edge Impulse
    """

    def setUp(self):
        # Suppress annoying message from request's socket
        # https://github.com/psf/requests/issues/3912
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning
        )

    def test_incorrect_api_key(self):
        # Upload dummy dataset
        dataset = util.create_dataset_good_csv()
        ids = upload_dataset(dataset)

        # Try to download samples with incorrect API key
        original_key = ei.API_KEY
        try:
            ei.API_KEY = "some_invalid_key"
            with self.assertRaises(UnauthorizedException):
                _ = download_samples_by_ids(
                    sample_ids=ids,
                    timeout_sec=TIMEOUT,
                )

        # Raise any exceptions and delete files from project
        except Exception as e:
            raise e
        finally:
            ei.API_KEY = original_key
            resp = delete_all_samples(
                timeout_sec=TIMEOUT,
            )
            if resp is None:
                logging.warning("Could not delete samples from project")

    def test_call_with_api_key(self):
        # Upload basic dataset
        dataset = util.create_dataset_good_csv()
        ids = upload_dataset(dataset)

        # Override incorrect API key with the correct key
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"

        # Try to download samples with manually specified API key
        try:
            dl_samples = download_samples_by_ids(
                sample_ids=ids,
                api_key=original_key,
                timeout_sec=TIMEOUT,
            )

            # Check that data exists
            for dl_sample in dl_samples:
                self.assertIsNotNone(dl_sample.data)
                self.assertNotEquals(dl_sample.metadata, "")

        # Raise any exceptions and delete files from project
        except Exception as e:
            raise e
        finally:
            ei.API_KEY = original_key
            resp = delete_all_samples(
                timeout_sec=TIMEOUT,
            )
            if resp is None:
                logging.warning("Could not delete samples from project")

    def test_non_int_ids(self):
        # Upload dummy dataset
        dataset = util.create_dataset_good_csv()

        # Try to download samples with non-integer IDs
        try:
            # Upload samples
            _ = upload_dataset(dataset)

            # Try to download samples with non-integer IDs
            with self.assertRaises(TypeError):
                _ = download_samples_by_ids(
                    sample_ids=["1", "2", "3"],
                    timeout_sec=TIMEOUT,
                )

        # Raise any exceptions and delete files from project
        except Exception as e:
            raise e
        finally:
            resp = delete_all_samples(
                timeout_sec=TIMEOUT,
            )
            if resp is None:
                logging.warning("Could not delete samples from project")

    def test_download_bad_ids(self):
        # Generate samples
        dataset = util.create_dataset_images()

        # Check for files, upload, check again. Always delete files from project when done.
        try:
            # Upload samples
            ids = upload_dataset(dataset)

            # Generate non-existent IDs
            bad_ids = []
            for i in range(len(ids)):
                bad_id = 1
                while bad_id in ids or bad_id in bad_ids:
                    bad_id = random.randint(1, 10000)
                bad_ids.append(bad_id)
            logging.info(f"Bad IDs: {bad_ids}")

            # Try to download non-existent samples
            dl_samples = download_samples_by_ids(
                sample_ids=bad_ids,
                timeout_sec=TIMEOUT,
                show_progress=True,
            )

            # Check that no samples were downloaded
            self.assertEqual(len(dl_samples), 0)

        # Raise any exceptions and delete files from project
        except Exception as e:
            raise e
        finally:
            resp = delete_all_samples(
                timeout_sec=TIMEOUT,
            )
            if resp is None:
                logging.warning("Could not delete samples from project")

    def test_download_files(self):
        # Generate samples
        datasets = util.create_all_good_datasets()

        # Upload samples
        for dataset in datasets:
            # Check for files, upload, check again. Always delete files from project when done.
            try:
                # Upload samples
                ids = upload_dataset(dataset)

                # Download samples
                dl_samples = download_samples_by_ids(
                    sample_ids=ids,
                    timeout_sec=TIMEOUT,
                )

                # Check that sample data and metadata exist
                self.assertGreater(len(dl_samples), 0)
                for dl_sample in dl_samples:
                    self.assertNotEqual(dl_sample.data, "")
                    self.assertNotEqual(dl_sample.metadata, "")

                # Check that the samples can be re-uploaded (for symmetry)
                resp = upload_samples(
                    dl_samples,
                    allow_duplicates=True,
                    timeout_sec=TIMEOUT,
                )

                # Check responses
                self.assertEqual(len(resp.successes), len(dataset))
                self.assertEqual(len(resp.fails), 0)
                for success in resp.successes:
                    self.assertIsNotNone(success.sample.sample_id)

                # Verify that 2 files with the same name are in the project
                for sample in dataset:
                    filename = os.path.splitext(sample.filename)[0]
                    infos = get_sample_ids(
                        filename=filename,
                        timeout_sec=TIMEOUT,
                    )
                    self.assertEqual(len(infos), 2)

            # Raise any exceptions and delete files from project
            except Exception as e:
                raise e
            finally:
                resp = delete_all_samples(
                    timeout_sec=TIMEOUT,
                )
                if resp is None:
                    logging.warning("Could not delete samples from project")
