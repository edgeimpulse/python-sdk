# ruff: noqa: D100, D101, D102, D103
import unittest
import resource
import logging

from tests.util import delete_all_samples, assert_uploaded_samples
from edgeimpulse import data, datasets

logging.getLogger().setLevel(logging.INFO)

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class TestUploadFiles(unittest.TestCase):
    def setUp(self):
        delete_all_samples()

    def test_upload_directory(self):
        res = data.upload_directory(
            directory="tests/sample_data/gestures",
            category="testing",
            metadata={"device": "phone"},
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(self, res.successes)

    def test_upload_directory_allow_duplicates(self):
        datasets.download_dataset("gestures")

        res = data.upload_directory(
            directory="datasets/gestures", allow_duplicates=True
        )
        self.assertEqual(len(res.successes), 113)

        assert_uploaded_samples(self, res.successes)

    def test_upload_directory_multi_label(self):
        res = data.upload_directory(directory="tests/sample_data/coffee")

        self.assertEqual(len(res.successes), 11)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(
            self,
            res.successes,
            check_meta=False,
            check_label=False,
            check_structured_labels=True,
        )

    def test_upload_directory_with_transform(self):
        # define a transform that will be called before file upload
        def transform(sample, file):
            sample.label = "human"

        res = data.upload_directory(
            directory="tests/sample_data/gestures", transform=transform
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(self, res.successes)

    def test_upload_directory_with_labels(self):
        # should auto detect the presence of a label file
        res = data.upload_directory(directory="tests/sample_data/dataset", batch_size=4)

        self.assertEqual(len(res.successes), 6)
        self.assertEqual(len(res.fails), 0)

        self.assertEqual(res.successes[0].sample.label, "background")
        assert_uploaded_samples(self, res.successes)

    def test_invalid_directory_path(self):
        with self.assertRaises(FileNotFoundError) as context:
            data.upload_directory(directory="tests/sample_data/dataset2")
        self.assertIn(
            "directory 'tests/sample_data/dataset2' not found.",
            str(context.exception),
        )

    def test_no_labels_file(self):
        with self.assertRaises(FileNotFoundError) as context:
            data.upload_exported_dataset(directory="tests/sample_data/")
        self.assertIn(
            "Labels file 'info.labels' not found in the specified directory.",
            str(context.exception),
        )

    # Test upload in batches
    def test_upload_directory_batches(self):
        res = data.upload_directory(
            directory="tests/sample_data/gestures", batch_size=10
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(self, res.successes)
