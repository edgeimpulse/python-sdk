import unittest
import resource
import logging
from tests.util import delete_all_samples, assert_uploaded_samples

from edgeimpulse.data._functions.upload_files import (
    upload_exported_dataset,
    upload_directory,
)

logging.getLogger().setLevel(logging.INFO)

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class TestUploadFiles(unittest.TestCase):
    def setUp(self):
        delete_all_samples()

    def test_upload_directory(self):
        res = upload_directory(
            directory="tests/sample_data/gestures",
            category="testing",
            metadata={"name": "jan"},
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(self, res.successes)

    def test_upload_directory_with_transform(self):
        # define a transform that will be called before file upload
        def transform(sample, file):
            sample.label = "human"

        res = upload_directory(
            directory="tests/sample_data/gestures", transform=transform
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        assert_uploaded_samples(self, res.successes)

    def test_upload_directory_with_labels(self):
        # should auto detect the presence of a label file
        res = upload_directory(directory="tests/sample_data/dataset")

        self.assertEqual(len(res.successes), 6)
        self.assertEqual(len(res.fails), 0)

        self.assertEqual(res.successes[0].sample.label, "background")
        assert_uploaded_samples(self, res.successes)

    def test_invalid_directory_path(self):
        with self.assertRaises(FileNotFoundError) as context:
            upload_directory(directory="tests/sample_data/dataset2")
        self.assertIn(
            "directory 'tests/sample_data/dataset2' not found.",
            str(context.exception),
        )

    def test_no_labels_file(self):
        with self.assertRaises(FileNotFoundError) as context:
            upload_exported_dataset(directory="tests/sample_data/")
        self.assertIn(
            "Labels file 'info.labels' not found in the specified directory.",
            str(context.exception),
        )
