# ruff: noqa: D100, D101, D102, D103
import edgeimpulse_api
import edgeimpulse as ei

import unittest

import logging
import pathlib
import os
import warnings

from edgeimpulse import util

# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for jobs to complete
JOB_TIMEOUT = 3600.0  # 60 min


def sample_model_path(model_fname):
    current_dir = pathlib.Path(__file__).parent.resolve()
    return os.path.join(current_dir, "sample_models", model_fname)


class TestProfile(unittest.TestCase):
    def setUp(self):
        # Suppress annoying message from request's socket
        # https://github.com/psf/requests/issues/3912
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning
        )

    def test_incorrect_api_key(self):
        # clobber config that's already been read from envvar
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"
        try:
            with self.assertRaises(edgeimpulse_api.exceptions.UnauthorizedException):
                _ = ei.model.profile(
                    model="some_path",
                    device="cortex-m4f-80mhz",
                )
        finally:
            # set global key back for other tests.
            ei.API_KEY = original_key

    def test_call_with_api_key(self):
        # Override incorrect API key with correct API key
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"
        try:
            profile_response = ei.model.profile(
                model=sample_model_path("gestures-i8.lite"),
                api_key=original_key,
                timeout_sec=JOB_TIMEOUT,
            )
            self.assertTrue(profile_response.success)
        finally:
            ei.API_KEY = original_key

    def test_zip_file_thats_not_a_saved_model(self):
        # TODO: should have a better error message here, see issue #6692
        with self.assertRaisesRegex(Exception, "No uploaded model yet"):
            _ = ei.model.profile(
                model=sample_model_path("random_zip_file.zip"),
                device="cortex-m4f-80mhz",
            )

    def test_invalid_path_for_tflite_file(self):
        with self.assertRaisesRegex(
            ei.exceptions.InvalidModelException, "No such file or directory"
        ):
            _ = ei.model.profile(
                model="/an/invalid/path",
                device="cortex-m4f-80mhz",
            )

    def test_invalid_device(self):
        with self.assertRaisesRegex(
            ei.exceptions.InvalidDeviceException, "Invalid device.*valid types are"
        ):
            _ = ei.model.profile(
                model=sample_model_path("gestures-i8.lite"),
                device="mortex-c4",
            )

    def test_timeout(self):
        # Test profile polling timeout
        with self.assertRaises(ei.exceptions.TimeoutException):
            _ = ei.model.profile(
                model=sample_model_path("gestures-i8.lite"),
                device="cortex-m4f-80mhz",
                timeout_sec=5.0,
            )

    def test_i8_model_profile(self):
        model_list = [
            "gestures-i8.lite",
            "temperature-regression.i8.lite",
            "fomo.96x96.i8q.lite",
            "responding-to-your-voice-tutorial.i8.lite",
        ]
        for i in model_list:
            with self.subTest(i=i):
                profile_response = ei.model.profile(
                    model=sample_model_path(i),
                    device="cortex-m4f-80mhz",
                    timeout_sec=JOB_TIMEOUT,
                )

                # check call was successful
                self.assertTrue(profile_response.success)

                # check that all ops are supported on MCU
                self.assertTrue(
                    profile_response.model.profile_info.int8.is_supported_on_mcu
                )

                # check that inference time is returned
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.time_per_inference_ms
                )

                # check we have info re: memory usage for both eon
                # and tf lite
                self.assertIsNotNone(profile_response.model.profile_info.int8.memory)
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.eon
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.eon.ram
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.eon.rom
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.tflite
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.tflite.ram
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.int8.memory.tflite.rom
                )

    def test_f32_model_profile(self):
        model_list = [
            "temperature-regression.f32.lite",
            "fomo.96x96.f32.lite",
            "responding-to-your-voice-tutorial.f32.lite",
            "a_zipped_keras_saved_model.zip",
            "saved_model.zip",
            "a_keras_saved_model",
            "saved_model",  # we include this since there is special handling for this dir name
        ]
        for i in model_list:
            with self.subTest(i=i):
                profile_response = ei.model.profile(
                    model=sample_model_path(i),
                    # had to change this as these models all have None for time_per_inference_ms
                    device="cortex-m7-216mhz",
                    timeout_sec=JOB_TIMEOUT,
                )

                # check call was successful
                self.assertTrue(profile_response.success)

                # check that all ops are supported on MCU
                self.assertTrue(
                    profile_response.model.profile_info.float32.is_supported_on_mcu
                )

                # check that inference time is returned
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.time_per_inference_ms
                )

                # check we have info re: memory usage for both eon
                # and tf lite
                self.assertIsNotNone(profile_response.model.profile_info.float32.memory)
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.eon
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.eon.ram
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.eon.rom
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.tflite
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.tflite.ram
                )
                self.assertIsNotNone(
                    profile_response.model.profile_info.float32.memory.tflite.rom
                )

    def test_profile_big_models(self):
        model_list = [
            "object-detection-tutorial.f32.lite",
            "object-detection-tutorial.i8.lite",
            "object-detection-yolov5.f32.lite",
            "object-detection-ti-yolox.f32.lite",
        ]
        for i in model_list:
            with self.subTest(i=i):
                profile_response = ei.model.profile(
                    model=sample_model_path(i),
                    device="raspberry-pi-4",
                    timeout_sec=JOB_TIMEOUT,
                )
                self.assertTrue(profile_response.success)
                if ".i8." in i:
                    self.assertFalse(
                        profile_response.model.profile_info.int8.is_supported_on_mcu
                    )
                    self.assertIsNotNone(
                        profile_response.model.profile_info.int8.tflite_file_size_bytes
                    )
                else:
                    self.assertFalse(
                        profile_response.model.profile_info.float32.is_supported_on_mcu
                    )
                    self.assertIsNotNone(
                        profile_response.model.profile_info.float32.tflite_file_size_bytes
                    )

    def test_profile_multiple(self):
        profile_response = ei.model.profile(
            model=sample_model_path("gestures-i8.lite"),
            timeout_sec=JOB_TIMEOUT,
        )
        self.assertTrue(profile_response.success)
        self.assertTrue(profile_response.model.profile_info.table)

    def test_profile_bytes(self):
        """Should be able to pass model bytes, not just a path."""
        model_path = sample_model_path("gestures-i8.lite")
        with open(model_path, "rb") as model_file:
            profile_response = ei.model.profile(
                model=model_file.read(),
                timeout_sec=JOB_TIMEOUT,
            )
        self.assertTrue(profile_response.success)
        self.assertTrue(profile_response.model.profile_info.table)

    @unittest.skipUnless(
        util.tensorflow_installed(), "Test requires TensorFlow but it was not available"
    )
    def test_profile_keras_model(self):
        # build minimal keras model
        import tensorflow as tf

        tf.keras.utils.set_random_seed(123)
        inputs = tf.keras.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(4)(inputs)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        profile_response = ei.model.profile(
            model=keras_model,
            timeout_sec=JOB_TIMEOUT,
        )
        # check call was successful
        self.assertTrue(profile_response.success)

        # check that all ops are supported on MCU
        self.assertTrue(profile_response.model.profile_info.float32.is_supported_on_mcu)

        # check that inference time is returned
        self.assertIsNotNone(
            profile_response.model.profile_info.float32.time_per_inference_ms
        )

        # check we have info re: memory usage for both eon
        # and tf lite
        self.assertIsNotNone(profile_response.model.profile_info.float32.memory)
        self.assertIsNotNone(profile_response.model.profile_info.float32.memory.eon)
        self.assertIsNotNone(profile_response.model.profile_info.float32.memory.eon.ram)
        self.assertIsNotNone(profile_response.model.profile_info.float32.memory.eon.rom)
        self.assertIsNotNone(profile_response.model.profile_info.float32.memory.tflite)
        self.assertIsNotNone(
            profile_response.model.profile_info.float32.memory.tflite.ram
        )
        self.assertIsNotNone(
            profile_response.model.profile_info.float32.memory.tflite.rom
        )
