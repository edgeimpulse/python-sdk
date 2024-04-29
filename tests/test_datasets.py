# ruff: noqa: D100, D101, D102, D103
import unittest
import logging
import os
from edgeimpulse.datasets import download_dataset

logging.getLogger().setLevel(logging.INFO)


class TestDatasets(unittest.TestCase):
    def test_download_dataset(self):
        download_dataset("gestures")
        self.assertTrue(os.path.exists("datasets/gestures"))

    def test_download_dataset_tar_gz(self):
        download_dataset("visual-xs")
        self.assertTrue(os.path.exists("datasets/gestures"))

    def test_download_non_existing_dataset(self):
        with self.assertRaises(ValueError):
            download_dataset("faucets2")
