import unittest

from edgeimpulse.model.input_type import AudioInput, TimeSeriesInput
from edgeimpulse.model.output_type import Classification, Regression, ObjectDetection
from edgeimpulse.model.model_info import ModelInfo

from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_input import (
    DeployPretrainedModelRequestModelInfoInput,
)
from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_model import (
    DeployPretrainedModelRequestModelInfoModel,
)


class TestModelInfo(unittest.TestCase):
    def test_audio_classification_model_info(self):
        actual = ModelInfo(
            input=AudioInput(frequency_hz=12000),
            model=Classification(labels=["a", "b", "c"]),
        )
        expected = {
            "input": DeployPretrainedModelRequestModelInfoInput.from_dict(
                {"inputType": "audio", "frequencyHz": 12000}
            ),
            "model": DeployPretrainedModelRequestModelInfoModel.from_dict(
                {
                    "modelType": "classification",
                    "labels": ["a", "b", "c"],
                }
            ),
        }
        self.assertDictEqual(actual, expected)

    def test_time_series_regression_model_info(self):
        actual = ModelInfo(
            input=TimeSeriesInput(frequency_hz=5000, windowlength_ms=100),
            model=Regression(),
        )
        expected = {
            "input": DeployPretrainedModelRequestModelInfoInput.from_dict(
                {
                    "inputType": "time-series",
                    "frequencyHz": 5000,
                    "windowLengthMs": 100,
                }
            ),
            "model": DeployPretrainedModelRequestModelInfoModel.from_dict(
                {"modelType": "regression"}
            ),
        }
        self.assertDictEqual(actual, expected)

    def test_object_detection_model_info(self):
        actual = ModelInfo(
            model=ObjectDetection(
                labels=["d", "e", "f"], last_layer="yolov5", minimum_confidence=0.3
            )
        )
        expected = {
            "input": DeployPretrainedModelRequestModelInfoInput.from_dict(
                {"inputType": "other"}
            ),
            "model": DeployPretrainedModelRequestModelInfoModel.from_dict(
                {
                    "modelType": "object-detection",
                    "labels": ["d", "e", "f"],
                    "lastLayer": "yolov5",
                    "minimumConfidence": 0.3,
                }
            ),
        }
        self.assertDictEqual(actual, expected)
