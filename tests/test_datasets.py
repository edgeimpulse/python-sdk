# ruff: noqa: D100, D101, D102, D103
import unittest
import logging
import os
import edgeimpulse as ei

logging.getLogger().setLevel(logging.INFO)


class TestDatasets(unittest.TestCase):
    def test_download_dataset(self):
        ei.datasets.download_dataset("gestures")
        self.assertTrue(os.path.exists("datasets/gestures"))

    def test_download_dataset_tar_gz(self):
        ei.datasets.download_dataset("visual-xs")
        self.assertTrue(os.path.exists("datasets/gestures"))

    def test_download_non_existing_dataset(self):
        with self.assertRaises(ValueError):
            ei.datasets.download_dataset("faucets2")
