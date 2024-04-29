# ruff: noqa: D100, D101, D102, D103
import unittest
from pathlib import Path
import edgeimpulse as ei
from edgeimpulse import util
import zipfile
import tempfile
from edgeimpulse import EdgeImpulseApi


def sample_model_path(model_fname):
    current_dir = Path(__file__).parent.resolve()
    return Path(current_dir, "sample_models", model_fname)


def valid_saved_model_zip(fname):
    zip_file = zipfile.ZipFile(fname)
    if zip_file.testzip() is not None:
        return False
    file_in_expected_location = "saved_model/saved_model.pb" in zip_file.namelist()
    return file_in_expected_location


class TestDefaultProjectIdFor(unittest.TestCase):
    def test_invalid_auth_type(self):
        with self.assertRaises(util.InvalidAuthTypeException):
            (
                util.default_project_id_for(
                    util.configure_generic_client("jwt_key_string", "jwt")
                ),
            )


class TestEasyApi(unittest.TestCase):
    def test_easy_api_key(self):
        api = EdgeImpulseApi(host=ei.API_ENDPOINT)
        api.authenticate(key=ei.API_KEY)
        pr_id = api.default_project_id()
        api.devices.list_devices(pr_id)

    def test_easy_api_jwt(self):
        api = EdgeImpulseApi(host=ei.API_ENDPOINT)
        if not ei.EI_USERNAME or not ei.EI_PASSWORD:
            raise Exception("Either EI_USERNAME or EI_PASSWORD isn't set")

        res = api.login.login({"username": ei.EI_USERNAME, "password": ei.EI_PASSWORD})

        api = EdgeImpulseApi(host=ei.API_ENDPOINT)
        api.authenticate(key=res.token, key_type="jwt_http")
        user = api.user.get_current_user()
        print(user)


class TestConfigureGenericClient(unittest.TestCase):
    def test_missing_api_key(self):
        with self.assertRaises(util.MissingApiKeyException):
            util.configure_generic_client(None)


class TestInspectModel(unittest.TestCase):
    def test_can_recognize_paths_and_binary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # Use onnx extension
            model_type, path = util.inspect_model("/tmp/something.onnx", tempdir)
            self.assertEqual(model_type, "onnx")
            self.assertEqual(path, "/tmp/something.onnx")
        with tempfile.TemporaryDirectory() as tempdir:
            # A path to a saved_model in a zip
            saved_model_path = sample_model_path("saved_model.zip")
            model_type, path = util.inspect_model(saved_model_path, tempdir)
            self.assertEqual(model_type, "saved_model")
            self.assertEqual(path, saved_model_path)
        with tempfile.TemporaryDirectory() as tempdir:
            # A path to a saved_model in a zip
            saved_model_path = sample_model_path("a_zipped_keras_saved_model.zip")
            model_type, path = util.inspect_model(saved_model_path, tempdir)
            self.assertEqual(model_type, "saved_model")
            self.assertEqual(path, saved_model_path)
        with tempfile.TemporaryDirectory() as tempdir:
            # A path to a saved_model directory called saved_model
            saved_model_path = sample_model_path("saved_model")
            model_type, path = util.inspect_model(saved_model_path, tempdir)
            self.assertEqual(model_type, "saved_model")
            self.assertEqual(path, f"{tempdir}/saved_model.zip")
            self.assertTrue(valid_saved_model_zip(path))
        with tempfile.TemporaryDirectory() as tempdir:
            # A path to a saved_model directory that isn't called saved_model
            # is repackaged in a directory called saved_model
            saved_model_path = sample_model_path("a_keras_saved_model")
            model_type, path = util.inspect_model(saved_model_path, tempdir)
            self.assertEqual(model_type, "saved_model")
            self.assertEqual(path, f"{tempdir}/saved_model.zip")
            self.assertTrue(valid_saved_model_zip(path))
        with tempfile.TemporaryDirectory() as tempdir:
            # Assume any other string is a tflite path
            model_type, path = util.inspect_model("some_string", tempdir)
            self.assertEqual(model_type, "tflite")
            self.assertEqual(path, "some_string")
        with tempfile.TemporaryDirectory() as tempdir:
            # If it's some random stuff assume it's tflite binary
            model_type, path = util.inspect_model(b"666", tempdir)
            self.assertEqual(model_type, "tflite")
            self.assertEqual(path, f"{tempdir}/model")
            with self.assertRaises(ei.exceptions.InvalidModelException):
                util.inspect_model(666, tempdir)

    @unittest.skipUnless(
        util.tensorflow_installed(),
        "Test requires TensorFlow, but it was not available",
    )
    def test_can_recognize_keras_model(self):
        # build minimal keras model
        import tensorflow as tf

        tf.keras.utils.set_random_seed(123)
        inputs = tf.keras.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(4)(inputs)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        with tempfile.TemporaryDirectory() as tempdir:
            model_type, path = util.inspect_model(keras_model, tempdir)

            self.assertEqual(model_type, "saved_model")
            self.assertEqual(path, f"{tempdir}/saved_model.zip")

    def test_can_recognize_onnx_model(self):
        import onnx

        # Load onnx model from file
        onnx_model_path = sample_model_path("accelerometer.onnx")
        onnx_model = onnx.load(onnx_model_path)

        # Check that model is saved to temp directory
        with tempfile.TemporaryDirectory() as tempdir:
            model_type, path = util.inspect_model(onnx_model, tempdir)
            self.assertEqual(model_type, "onnx")
            self.assertEqual(path, f"{tempdir}/model.onnx")


class TestInspectRepresentativeData(unittest.TestCase):
    def test_can_recognize_paths(self):
        # No path provided
        path = util.inspect_representative_data(None)
        self.assertEqual(path, None)

        # Numpy path
        path = util.inspect_representative_data("/tmp/data.npy")
        self.assertEqual(path, "/tmp/data.npy")

        # Some random path
        with self.assertRaisesRegex(Exception, "Unknown representative data file"):
            util.inspect_representative_data("boop")

        # Some random type
        with self.assertRaisesRegex(Exception, "Can't parse representative data"):
            util.inspect_representative_data(b"666")

    @unittest.skipUnless(
        util.numpy_installed(), "Test requires numpy but it was not available"
    )
    def test_can_recognize_numpy(self):
        import numpy as np

        array = np.zeros((2, 3, 4), dtype=np.float32)
        path = util.inspect_representative_data(array)

        # Path should be none since we didn't save it yet
        self.assertEqual(path, None)


class TestUtilHelperLists(unittest.TestCase):
    def test_get_profile_devices(self):
        client = util.configure_generic_client(
            key=ei.API_KEY,
            host=ei.API_ENDPOINT,
        )
        devices = util.get_profile_devices(client)
        #  assert stable ones and the fact we have some number, though don't peg to an exact value
        self.assertTrue("cortex-m4f-80mhz" in devices)
        self.assertTrue("jetson-nano" in devices)
        self.assertTrue(len(devices) > 10)

    def test_get_project_deploy_targets(self):
        client = util.configure_generic_client(
            key=ei.API_KEY,
            host=ei.API_ENDPOINT,
        )
        targets = util.get_project_deploy_targets(client)
        #  assert stable ones and the fact we have some number, though don't peg to an exact value
        self.assertTrue("zip" in targets)
        self.assertTrue("brickml" in targets)
        self.assertTrue(len(targets) > 10)
