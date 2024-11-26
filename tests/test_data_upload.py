# ruff: noqa: D100, D101, D102, D103
import unittest
import logging
import os
import warnings

import edgeimpulse as ei
from edgeimpulse import data

from edgeimpulse.data import Sample

from . import util

# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for uploading to complete
TIMEOUT = 3600.0  # 60 min


class TestDataUpload(unittest.TestCase):
    """Test upload features."""

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
        samples = util.create_dataset_images()
        resp = data.upload_samples(
            samples,
            allow_duplicates=False,
            timeout_sec=TIMEOUT,
        )

        # Restore original key
        ei.API_KEY = original_key

        # Check response
        self.assertEqual(len(resp.successes), 0)
        self.assertEqual(len(resp.fails), len(samples))
        for fail in resp.fails:
            self.assertEqual(fail.error, "Invalid API key")

    def test_call_with_api_key(self):
        # Override incorrect API key with the correct key
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"

        # Generate samples
        samples = util.create_dataset_images()

        # Try uploading samples
        try:
            # Upload samples
            resp = data.upload_samples(
                samples,
                allow_duplicates=False,
                api_key=original_key,
                timeout_sec=TIMEOUT,
            )

            # Check responses
            self.assertEqual(len(resp.successes), len(samples))
            self.assertEqual(len(resp.fails), 0)

        # Raise any exceptions, always delete files from project
        except Exception as e:
            raise e
        finally:
            ei.API_KEY = original_key
            for sample in samples:
                data.delete_samples_by_filename(
                    filename=os.path.splitext(sample.filename)[0],
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )

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
        samples = [Sample(**i) for i in dataset]

        # Upload garbage data and check response
        try:
            resp = data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )
            self.assertEqual(len(resp.successes), 0)
            self.assertEqual(len(resp.fails), len(samples))
            for fail in resp.fails:
                err = fail.error
                self.assertTrue(
                    err.startswith("Could not parse this CSV file")
                    or err.startswith("Unknown error")
                )

        # Raise any exceptions, always delete samples from project
        except Exception as e:
            raise e
        finally:
            for sample in dataset:
                data.delete_samples_by_filename(
                    filename=os.path.splitext(sample["filename"])[0],
                    category=sample["category"],
                    timeout_sec=TIMEOUT,
                )

    def test_upload_bad_csv(self):
        # Generate samples
        samples = util.create_dataset_bad_csv()

        # Upload garbage data and check response
        try:
            resp = data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )
            self.assertEqual(len(resp.successes), 0)
            self.assertEqual(len(resp.fails), len(samples))
            for fail in resp.fails:
                err = fail.error
                self.assertTrue(
                    err.startswith("Could not parse this CSV file")
                    or err.startswith("Unknown error")
                )

        # Raise any exceptions and always delete samples from project
        except Exception as e:
            raise e
        finally:
            resp = data.delete_all_samples(
                timeout_sec=TIMEOUT,
            )
            if resp is None:
                logging.warning("Could not delete samples from project")

    def test_upload_files(self):
        # Generate samples
        datasets = util.create_all_good_datasets()

        # Try uploading samples
        for samples in datasets:
            # Check for files, upload, check again. Always delete files from project when done.
            try:
                # Make sure there are no files in the project that match the filename
                for sample in samples:
                    filename = os.path.splitext(sample.filename)[0]
                    infos = data.get_sample_ids(
                        filename=filename,
                        category=sample.category,
                        timeout_sec=TIMEOUT,
                    )
                    self.assertEqual(len(infos), 0)

                # Upload samples
                resp = data.upload_samples(
                    samples,
                    allow_duplicates=False,
                    show_progress=True,
                    timeout_sec=TIMEOUT,
                )

                # Check responses
                self.assertEqual(len(resp.successes), len(samples))
                self.assertEqual(len(resp.fails), 0)
                for success in resp.successes:
                    self.assertIsNotNone(success.sample.sample_id)

                # Verify that the files are in the project
                for sample in samples:
                    filename = os.path.splitext(sample.filename)[0]
                    infos = data.get_sample_ids(
                        filename=filename,
                        category=sample.category,
                        timeout_sec=TIMEOUT,
                    )
                    self.assertEqual(len(infos), 1)

            # Raise any exceptions, always delete samples from project
            except Exception as e:
                raise e
            finally:
                resp = data.delete_all_samples(
                    timeout_sec=TIMEOUT,
                )
                if resp is None:
                    logging.warning("Could not delete samples from project")

    def test_upload_duplicates(self):
        # Define dataset
        samples = util.create_dataset_images()

        # Make sure there are no files in the project that match the filename
        try:
            for sample in samples:
                filename = os.path.splitext(sample.filename)[0]
                infos = data.get_sample_ids(
                    filename=filename,
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(infos), 0)
        except Exception as e:
            raise e
        finally:
            for sample in samples:
                data.delete_samples_by_filename(
                    filename=os.path.splitext(sample.filename)[0],
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )

        # Try uploading the same samples twice with "allow_duplicates"
        for _ in range(2):
            # Define dataset (to re-open the data files)
            samples = util.create_dataset_images()

            # Check responses
            try:
                # Upload samples
                resp = data.upload_samples(
                    samples, allow_duplicates=True, timeout_sec=TIMEOUT
                )

                # Check responses
                self.assertEqual(len(resp.successes), len(samples))
                self.assertEqual(len(resp.fails), 0)
                for success in resp.successes:
                    self.assertIsNotNone(success.sample.sample_id)

            # Raise any exceptions and delete files from project
            except Exception as e:
                for sample in samples:
                    data.delete_samples_by_filename(
                        filename=os.path.splitext(sample.filename)[0],
                        category=sample.category,
                        timeout_sec=TIMEOUT,
                    )
                raise e

        # Ensure exactly 2 files with the same name have been uploaded
        try:
            for sample in samples:
                filename = os.path.splitext(sample.filename)[0]
                infos = data.get_sample_ids(
                    filename=filename,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(infos), 2)
        except Exception as e:
            for sample in sample:
                data.delete_samples_by_filename(
                    filename=os.path.splitext(sample.filename)[0],
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )
            raise e

        # Try uploading the same samples twice without "allow_duplicates"
        try:
            # Upload samples without duplicates this time (should prevent uploading)
            samples = util.create_dataset_images()
            resp = data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )

            # Check responses
            self.assertEqual(len(resp.successes), 0)
            self.assertEqual(len(resp.fails), len(samples))
            for fail in resp.fails:
                self.assertTrue(
                    fail.error.startswith("An item with this hash already exists")
                )

        except Exception as e:
            raise e
        finally:
            for sample in samples:
                data.delete_samples_by_filename(
                    filename=os.path.splitext(sample.filename)[0],
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )

    def test_get_filename_with_bad_id(self):
        # Try an ID that does not exist
        ret_filename = data.get_filename_by_id(1)
        self.assertIsNone(ret_filename)

    def test_get_filename_with_good_id(self):
        # Define dataset
        samples = util.create_dataset_images()

        # Check for files, upload, check again. Always delete files from project when done.
        try:
            # Make sure there are no files in the project that match the filename
            for sample in samples:
                filename = os.path.splitext(sample.filename)[0]
                infos = data.get_sample_ids(
                    filename=filename,
                    category=sample.category,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(infos), 0)

            # Upload samples
            resp = data.upload_samples(
                samples, allow_duplicates=False, timeout_sec=TIMEOUT
            )

            # Check responses
            self.assertEqual(len(resp.successes), len(samples))
            self.assertEqual(len(resp.fails), 0)
            for success in resp.successes:
                self.assertIsNotNone(success.sample.sample_id)

            # Verify the returned filename is the same as the one given
            for sample in samples:
                filename = os.path.splitext(sample.filename)[0]
                infos = data.get_sample_ids(
                    filename=filename,
                    timeout_sec=TIMEOUT,
                )
                self.assertEqual(len(infos), 1)
                ret_filename = data.get_filename_by_id(infos[0].sample_id)
                self.assertEqual(ret_filename, filename)

        # Raise any exceptions, always delete samples from project
        except Exception as e:
            raise e
        finally:
            for category in ei.util.DATA_CATEGORIES:
                resp = data.delete_all_samples(
                    category=category,
                    timeout_sec=TIMEOUT,
                )
                if resp is None:
                    logging.warning("Could not delete samples from project")
