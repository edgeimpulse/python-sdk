# ruff: noqa: D100, D101, D102, D103
import unittest
import os
import shutil
from edgeimpulse import datasets

# logging.getLogger().setLevel(logging.INFO)


class TestDatasets(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("datasets", ignore_errors=True)

    def tearDown(self):
        shutil.rmtree("datasets", ignore_errors=True)

    def test_list_datasets(self):
        datasets.list_datasets()

    def test_download_dataset(self):
        datasets.download_dataset("gestures")
        self.assertTrue(os.path.exists("datasets/gestures"))

    def test_download_dataset_tar_gz(self):
        datasets.download_dataset("visual-xs")
        self.assertTrue(os.path.exists("datasets/visual-xs"))

    def test_download_non_existing_dataset(self):
        with self.assertRaises(ValueError):
            datasets.download_dataset("faucets2")
