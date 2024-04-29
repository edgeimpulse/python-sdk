# ruff: noqa: D100, D101, D102, D103
import pathlib
import os
import warnings

# TODO: switch to import edgeimpulse as ei
import edgeimpulse
import edgeimpulse as ei
import tempfile
import unittest
import zipfile
import re

from edgeimpulse import util
from edgeimpulse.model.input_type import ImageInput, AudioInput, TimeSeriesInput
from edgeimpulse.model.output_type import Classification, Regression, ObjectDetection
from edgeimpulse.model._functions.deploy import (
    _determine_deploy_type,
    _determine_output_type,
)
from edgeimpulse.exceptions import (
    InvalidEngineException,
    InvalidDeployParameterException,
    InvalidModelException,
    EdgeImpulseException,
)

from edgeimpulse_api.models.pretrained_model_tensor import PretrainedModelTensor

# just have logging enabled for dev
import logging

logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for jobs to complete
JOB_TIMEOUT = 1200.0  # 20 min


def sample_model_path(model_fname):
    current_dir = pathlib.Path(__file__).parent.resolve()
    return os.path.join(current_dir, "sample_models", model_fname)


def temp_filename():
    # TODO: refactor all uses of this method with
    #   with tempfile.TemporaryDirectory() as tmp_dir:
    #     output_filename = f"{tmp_dir}/test_xyz.zip"
    return os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names())
    )


class TestDetermineDeployType(unittest.TestCase):
    def test_unavailable_types(self):
        with self.assertRaises(InvalidDeployParameterException):
            _ = _determine_deploy_type(
                deploy_model_type="int8",
                representative_data_for_quantization=None,
                available_model_types=["float32"],
            )

    def test_no_representative_data(self):
        deploy_type = _determine_deploy_type(
            deploy_model_type=None,
            representative_data_for_quantization=None,
            available_model_types=["float32"],
        )
        self.assertEqual(deploy_type, "float32")

        deploy_type = _determine_deploy_type(
            deploy_model_type=None,
            representative_data_for_quantization=None,
            available_model_types=["int8"],
        )
        self.assertEqual(deploy_type, "int8")

        with self.assertRaises(InvalidDeployParameterException):
            deploy_type = _determine_deploy_type(
                deploy_model_type=None,
                representative_data_for_quantization=None,
                available_model_types=["akida"],
            )

    def test_representative_data(self):
        deploy_type = _determine_deploy_type(
            deploy_model_type="float32",
            representative_data_for_quantization=[],
            available_model_types=["int8", "float32"],
        )
        self.assertEqual(deploy_type, "float32")

        deploy_type = _determine_deploy_type(
            deploy_model_type=None,
            representative_data_for_quantization=[],
            available_model_types=["int8"],
        )
        self.assertEqual(deploy_type, "int8")

        with self.assertRaises(InvalidDeployParameterException):
            deploy_type = _determine_deploy_type(
                deploy_model_type=None,
                representative_data_for_quantization=[],
                available_model_types=["float32"],
            )


class DetermineOutputType(unittest.TestCase):
    classifier_output = PretrainedModelTensor.from_dict(
        {"dataType": "int8", "name": "x", "shape": [1, 3]}
    )
    big_classifier_output = PretrainedModelTensor.from_dict(
        {"dataType": "int8", "name": "x", "shape": [1, 10]}
    )
    regression_output = PretrainedModelTensor.from_dict(
        {"dataType": "int8", "name": "x", "shape": [1, 1]}
    )
    extradimensional_output = PretrainedModelTensor.from_dict(
        {"dataType": "int8", "name": "x", "shape": [1, 1, 3]}
    )

    def test_valid_output_numbers(self):
        # Normal classification, correct number of outputs
        output_type = _determine_output_type(
            model_output_type=Classification(labels=["1", "2", "3"]),
            outputs=[self.classifier_output],
        )
        self.assertEqual(output_type["labels"], ["1", "2", "3"])
        # Normal regression, correct number of outputs
        output_type = _determine_output_type(
            model_output_type=Regression(),
            outputs=[self.regression_output],
        )

    def test_invalid_output_numbers(self):
        # Too many outputs
        with self.assertRaises(InvalidModelException):
            _ = _determine_output_type(
                model_output_type=Classification(labels=["1", "2", "3"]),
                outputs=[self.classifier_output, self.regression_output],
            )
        # Too many output dimensions
        with self.assertRaises(InvalidModelException):
            _ = _determine_output_type(
                model_output_type=Classification(labels=["1", "2", "3"]),
                outputs=[self.extradimensional_output],
            )
        # Too many output neurons
        with self.assertRaises(InvalidDeployParameterException):
            _ = _determine_output_type(
                model_output_type=Classification(labels=["1", "2", "3"]),
                outputs=[self.big_classifier_output],
            )
        # Non scalar regression output
        with self.assertRaises(InvalidModelException):
            _ = _determine_output_type(
                model_output_type=Regression(),
                outputs=[self.classifier_output],
            )

    def test_assigns_classifier_labels(self):
        # If no labels provided it assigns them
        output_type = _determine_output_type(
            model_output_type=Classification(), outputs=[self.classifier_output]
        )
        self.assertEqual(output_type["labels"], ["0", "1", "2"])


class TestDeploy(unittest.TestCase):
    def setUp(self):
        # Suppress annoying message from request's socket
        # https://github.com/psf/requests/issues/3912
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning
        )

    def verify_valid_zip_file(self, fname=None, data=None):
        # verify zip; will throw exception on ZipFile creation
        # and return None if zip has valid checksums etc
        if fname:
            if zipfile.ZipFile(fname).testzip() is not None:
                raise Exception("Bad zipfile created by deploy")

        if data:
            if zipfile.ZipFile(data).testzip() is not None:
                raise Exception("Bad zip data created by deploy")

        if fname and data:
            # Rewind the stream
            data.seek(0)
            with open(fname, "rb") as f:
                contents = f.read()
                self.assertEqual(
                    data.read(), contents, "Zip in memory and on disk are not the same"
                )

    def get_only_file(self, dirname):
        """Return the only file in a directory."""
        files = os.listdir(dirname)
        self.assertEqual(len(files), 1)
        return os.path.join(dirname, files[0])

    # some common scenarios covered by TestProfile...
    # def test_incorrect_api_key(self):
    # def test_invalid_path_for_tflite_file(self):

    def test_timeout(self):
        # Test deploy polling timeout
        with tempfile.TemporaryDirectory() as dirname:
            with self.assertRaises(ei.exceptions.TimeoutException):
                _ = edgeimpulse.model.deploy(
                    model=sample_model_path("fan-v3.f32.lite"),
                    model_output_type=Classification(
                        labels=["class0", "class1", "class2", "class3"]
                    ),
                    output_directory=dirname,
                    deploy_model_type="float32",
                    timeout_sec=5.0,
                )

    def test_f32_classification_model(self):
        # TODO: add URL to fan-v3 public project if there is one
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fan-v3.f32.lite"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_f32_classification_with_no_labels(self):
        # TODO: add URL to fan-v3 public project if there is one
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fan-v3.f32.lite"),
                model_output_type=Classification(),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_f32_classification_with_incorrect_labels(self):
        # TODO: add URL to fan-v3 public project if there is one
        with self.assertRaises(InvalidDeployParameterException):
            with tempfile.TemporaryDirectory() as dirname:
                _ = edgeimpulse.model.deploy(
                    model=sample_model_path("fan-v3.f32.lite"),
                    model_output_type=Classification(labels=["notenough"]),
                    output_directory=dirname,
                    deploy_model_type="float32",
                    timeout_sec=JOB_TIMEOUT,
                )

    def test_f32_classification_model_using_eon(self):
        # TODO: add URL to fan-v3 public project if there is one
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fan-v3.f32.lite"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                engine="tflite-eon",
                timeout_sec=JOB_TIMEOUT,
            )
            filename = self.get_only_file(dirname)
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: add a function version of this test to all deploy tests?
            zip_filelist = zipfile.ZipFile(filename).namelist()
            model_files = 0
            for filename in zip_filelist:
                if re.search(
                    "tflite_learn_[0-9]+_compiled.(h|c(pp)?)$", filename
                ) or filename.endswith("ops_define.h"):
                    model_files += 1
            self.assertTrue(model_files == 3)

    def test_f32_regression_model(self):
        # temperature-regression.f32.lite is from https://studio.edgeimpulse.com/public/17972/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.f32.lite"),
                model_output_type=Regression(),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_f32_regression_model_time_series(self):
        # temperature-regression.f32.lite is from https://studio.edgeimpulse.com/public/17972/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.f32.lite"),
                model_output_type=Regression(),
                model_input_type=TimeSeriesInput(frequency_hz=4, windowlength_ms=2000),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_i8_regression_model(self):
        # temperature-regression.i8.lite is from https://studio.edgeimpulse.com/public/17972/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.i8.lite"),
                model_output_type=Regression(),
                output_directory=dirname,
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_i8_expect_float32_exception(self):
        # temperature-regression.i8.lite is from https://studio.edgeimpulse.com/public/17972/latest
        with self.assertRaises(InvalidDeployParameterException):
            with tempfile.TemporaryDirectory() as dirname:
                _ = edgeimpulse.model.deploy(
                    model=sample_model_path("temperature-regression.i8.lite"),
                    model_output_type=Regression(),
                    output_directory=dirname,
                    deploy_model_type="float32",
                    timeout_sec=JOB_TIMEOUT,
                )

    def test_i8_classification_model(self):
        # TODO: (double check) gestures-i8.lite is from https://studio.edgeimpulse.com/public/14299/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("gestures-i8.lite"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                output_directory=dirname,
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_fomo_f32_object_detection_model(self):
        # TODO: (double check) fomo.96x96.f32.lite is from https://studio.edgeimpulse.com/public/89078/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fomo.96x96.f32.lite"),
                model_output_type=ObjectDetection(
                    labels=["beer", "can"], last_layer="fomo", minimum_confidence=0.3
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_yolov5_f32_object_detection_model(self):
        # object-detection-yolov5.f32.lite is from https://studio.edgeimpulse.com/public/198327/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("object-detection-yolov5.f32.lite"),
                model_output_type=ObjectDetection(
                    labels=["beer", "can"], last_layer="fomo", minimum_confidence=0.3
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                engine="tflite",
                timeout_sec=JOB_TIMEOUT,
            )
            filename = self.get_only_file(dirname)
            self.verify_valid_zip_file(filename, model)
            zip_filelist = zipfile.ZipFile(filename).namelist()
            model_files = 0
            for filename in zip_filelist:
                if re.search(
                    "tflite_learn_[0-9]+.(h|c(pp)?)$", filename
                ) or filename.endswith("tflite-resolver.h"):
                    print(filename)
                    model_files += 1
            self.assertTrue(model_files == 3)

    def test_yolox_f32_object_detection_model(self):
        # object-detection-ti-yolox.f32.lite is from https://studio.edgeimpulse.com/public/198327/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("object-detection-ti-yolox.f32.lite"),
                model_output_type=ObjectDetection(
                    labels=["coffee", "lamp"],
                    last_layer="yolox",
                    minimum_confidence=0.3,
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                engine="tflite",
                timeout_sec=JOB_TIMEOUT,
            )
            filename = self.get_only_file(dirname)
            self.verify_valid_zip_file(filename, model)
            zip_filelist = zipfile.ZipFile(filename).namelist()
            model_files = 0
            for filename in zip_filelist:
                if re.search(
                    "tflite_learn_[0-9]+.(h|c(pp)?)$", filename
                ) or filename.endswith("tflite-resolver.h"):
                    model_files += 1
            self.assertTrue(model_files == 3)

    def test_mobilenetssd_i8_model(self):
        # object-detection-tutorial.i8.lite is from https://studio.edgeimpulse.com/public/25483/latest
        model = edgeimpulse.model.deploy(
            model=sample_model_path("object-detection-tutorial.i8.lite"),
            model_output_type=ObjectDetection(
                labels=["lamp", "coffee"],
                last_layer="mobilenet-ssd",
                minimum_confidence=0.3,
            ),
            model_input_type=ImageInput(),
            deploy_model_type="int8",
            timeout_sec=JOB_TIMEOUT,
        )
        self.verify_valid_zip_file(None, model)

    def test_mobilenetssd_f32_model(self):
        # object-detection-tutorial.f32.lite is from https://studio.edgeimpulse.com/public/25483/latest
        model = edgeimpulse.model.deploy(
            model=sample_model_path("object-detection-tutorial.f32.lite"),
            model_output_type=ObjectDetection(
                labels=["lamp", "coffee"],
                last_layer="mobilenet-ssd",
                minimum_confidence=0.3,
            ),
            model_input_type=ImageInput(),
            deploy_model_type="float32",
            timeout_sec=JOB_TIMEOUT,
        )
        self.verify_valid_zip_file(None, model)

    def test_fomo_i8_object_detection_model(self):
        # TODO: (double check) fomo.96x96.i8q.lite is from https://studio.edgeimpulse.com/public/89078/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fomo.96x96.i8q.lite"),
                model_output_type=ObjectDetection(
                    labels=["beer", "can"], last_layer="fomo", minimum_confidence=0.3
                ),
                model_input_type=ImageInput(),
                output_directory=dirname,
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_scaling_range_255(self):
        # TODO: (double check) fomo.96x96.i8q.lite is from https://studio.edgeimpulse.com/public/89078/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fomo.96x96.i8q.lite"),
                model_output_type=ObjectDetection(
                    labels=["beer", "can"], last_layer="fomo", minimum_confidence=0.3
                ),
                model_input_type=ImageInput(scaling_range="0..255"),
                output_directory=dirname,
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_scaling_range_torch(self):
        # TODO: (double check) fomo.96x96.i8q.lite is from https://studio.edgeimpulse.com/public/89078/latest
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("fomo.96x96.i8q.lite"),
                model_output_type=ObjectDetection(
                    labels=["beer", "can"], last_layer="fomo", minimum_confidence=0.3
                ),
                model_input_type=ImageInput(scaling_range="torch"),
                output_directory=dirname,
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)

    def test_audio_i8_model(self):
        # responding-to-your-voice-tutorial.i8.lite is from https://studio.edgeimpulse.com/public/14225/latest
        model = edgeimpulse.model.deploy(
            model=sample_model_path("responding-to-your-voice-tutorial.i8.lite"),
            model_output_type=Classification(
                labels=["hello world", "noise", "unknown"]
            ),
            model_input_type=AudioInput(frequency_hz=16000),
            deploy_model_type="int8",
        )
        self.verify_valid_zip_file(None, model)

    def test_audio_f32_model(self):
        # responding-to-your-voice-tutorial.f32.lite is from https://studio.edgeimpulse.com/public/14225/latest
        model = edgeimpulse.model.deploy(
            model=sample_model_path("responding-to-your-voice-tutorial.f32.lite"),
            model_output_type=Classification(
                labels=["hello world", "noise", "unknown"]
            ),
            model_input_type=AudioInput(frequency_hz=16000),
            deploy_model_type="float32",
            timeout_sec=JOB_TIMEOUT,
        )
        self.verify_valid_zip_file(None, model)

    def test_zip_file_thats_not_a_saved_model(self):
        # TODO: should have a better error message here, see issue #6692
        with self.assertRaisesRegex(
            edgeimpulse.exceptions.EdgeImpulseException,
            "get_pretrained_model_info did not return model details",
        ):
            edgeimpulse.model.deploy(
                model=sample_model_path("random_zip_file.zip"),
                model_output_type=Regression(),
                timeout_sec=JOB_TIMEOUT,
            )

    def test_invalid_deploy_target(self):
        with self.assertRaisesRegex(
            ei.exceptions.InvalidTargetException,
            r"deploy_target: \[some_invalid_deploy_target\] not in",
        ):
            edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.f32.lite"),
                model_output_type=Regression(),
                deploy_target="some_invalid_deploy_target",
                timeout_sec=JOB_TIMEOUT,
            )

    def test_invalid_engine(self):
        with self.assertRaises(InvalidEngineException):
            edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.f32.lite"),
                model_output_type=Regression(),
                engine="sdfsdf",
                timeout_sec=JOB_TIMEOUT,
            )

    def test_call_with_api_key(self):
        # Override incorrect API key with correct API key
        original_key = ei.API_KEY
        ei.API_KEY = "some_invalid_key"
        try:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("temperature-regression.f32.lite"),
                model_output_type=Regression(),
                api_key=original_key,
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(None, model)
        finally:
            ei.API_KEY = original_key

    # TODO: for now saved_model.zip is a model made in a seperate script with
    #       the standard zip conversion we do elsewhere in studio, e.g.
    # import tensorflow as tf
    # import shutil, os
    # saved_model_path = "/tmp/saved_model"
    # inputs = tf.keras.Input(shape=(3,))
    # outputs = tf.keras.layers.Dense(4)(inputs)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.save(saved_model_path, save_format='tf')
    # shutil.make_archive(saved_model_path,
    # 	    'zip',
    # 	    root_dir=os.path.dirname(saved_model_path),
    # 	    base_dir='saved_model')
    # representative_data =  np.random.random((100, 3)).astype(np.float32)
    # np.save("saved_model-representative.npy",representative_data)

    def test_keras_saved_model_f32_by_zip(self):
        # model is float32 and no representative_data_for_quantization
        # provided => we get back a float32 model.
        # since we have special handling for files named "saved_model" we
        # check variants
        for zip_fname in ["a_zipped_keras_saved_model.zip", "saved_model.zip"]:
            with tempfile.TemporaryDirectory() as dirname:
                model = edgeimpulse.model.deploy(
                    model=sample_model_path(zip_fname),
                    model_output_type=Classification(
                        labels=["class0", "class1", "class2", "class3"]
                    ),
                    output_directory=dirname,
                    timeout_sec=JOB_TIMEOUT,
                )
                self.verify_valid_zip_file(self.get_only_file(dirname), model)
                # TODO: what else can we validate here?

    def test_keras_saved_model_f32_by_directory(self):
        # model is float32 and no representative_data_for_quantization
        # provided => we get back a float32 model.
        # since we have special handling for directories named "saved_model" we
        # check variants.
        for base_dir_name in ["a_keras_saved_model", "saved_model"]:
            with tempfile.TemporaryDirectory() as dirname:
                model = edgeimpulse.model.deploy(
                    model=sample_model_path(base_dir_name),
                    model_output_type=Classification(
                        labels=["class0", "class1", "class2", "class3"]
                    ),
                    output_directory=dirname,
                    timeout_sec=JOB_TIMEOUT,
                )
                self.verify_valid_zip_file(self.get_only_file(dirname), model)
                # TODO: what else can we validate here?

    def test_keras_saved_model_i8_with_representative_data_for_quantization_on_disk(
        self,
    ):
        # model is float32 and representative_data_for_quantization
        # provided => we get back a int8 model
        with tempfile.TemporaryDirectory() as dirname:
            path_to_representative_data_on_disk = sample_model_path(
                "saved_model-representative.npy"
            )
            model = edgeimpulse.model.deploy(
                model=sample_model_path("saved_model.zip"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                representative_data_for_quantization=path_to_representative_data_on_disk,
                output_directory=dirname,
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: what else can we validate here?

    @unittest.skipUnless(
        util.numpy_installed(), "Test requires numpy, but it was not available"
    )
    def test_keras_saved_model_i8_with_representative_data_for_quantization_in_memory(
        self,
    ):
        # model is float32 and representative_data_for_quantization
        # provided => we get back a int8 model
        import numpy as np

        with tempfile.TemporaryDirectory() as dirname:
            representative_data_in_memory = np.load(
                sample_model_path("saved_model-representative.npy")
            )
            model = edgeimpulse.model.deploy(
                model=sample_model_path("saved_model.zip"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                representative_data_for_quantization=representative_data_in_memory,
                output_directory=dirname,
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: what else can we validate here?

    @unittest.skip("Representative data shape validation is currently not working")
    def test_incorrect_shaped_representative_data_for_quantization(self):
        # model is float32 and representative_data_for_quantization
        # provided => we get back a int8 model
        # TODO: Re-enable once validation is added https://github.com/edgeimpulse/edgeimpulse/issues/6626
        with self.assertRaises(EdgeImpulseException):
            edgeimpulse.model.deploy(
                model=sample_model_path("saved_model.zip"),
                model_output_type=Classification(
                    labels=["class0", "class1", "class2", "class3"]
                ),
                representative_data_for_quantization=sample_model_path(
                    "saved_model-representative.wrong_shape.npy"
                ),
                timeout_sec=JOB_TIMEOUT,
            )

    def test_onnx_model_f32(self):
        # model is float32 and no representative_data_for_quantization
        # provided => we get back a float32 model
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("accelerometer.onnx"),
                model_output_type=Classification(
                    labels=["idle", "snake", "updown", "wave"]
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: what else can we validate here?

    def test_onnx_model_i8_without_representative_data_for_quantization(self):
        with self.assertRaises(InvalidDeployParameterException):
            edgeimpulse.model.deploy(
                model=sample_model_path("accelerometer.onnx"),
                model_output_type=Classification(
                    labels=["idle", "snake", "updown", "wave"]
                ),
                deploy_model_type="int8",
                timeout_sec=JOB_TIMEOUT,
            )
            # TODO: Check error message was propagated through exception

    def test_onnx_model_i8_with_representative_data_for_quantization(self):
        # model is float32 and representative_data_for_quantization
        # provided => we get back a int8 model
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=sample_model_path("accelerometer.onnx"),
                model_output_type=Classification(
                    labels=["idle", "snake", "updown", "wave"]
                ),
                representative_data_for_quantization=sample_model_path(
                    "accelerometer-representative.npy"
                ),
                output_directory=dirname,
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: validate was int8ified?

    def test_onnx_modelproto_f32(
        self,
    ):
        import onnx

        # Load onnx model from file
        onnx_model_path = sample_model_path("accelerometer.onnx")
        onnx_model = onnx.load(onnx_model_path)

        # model is float32 and no representative_data_for_quantization
        # provided => we get back a float32 model
        with tempfile.TemporaryDirectory() as dirname:
            model = edgeimpulse.model.deploy(
                model=onnx_model,
                model_output_type=Classification(
                    labels=["idle", "snake", "updown", "wave"]
                ),
                output_directory=dirname,
                deploy_model_type="float32",
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
            # TODO: what else can we validate here?
