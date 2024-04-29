# ruff: noqa:  D100, D101
from typing import Optional, List, Union, Literal


class Classification(dict):
    def __init__(self, labels: Optional[List[str]] = None):
        """Describe a classifier output with an optional list of label names.

        If no list is provided then numeric labels will be assigned according to the
        order of outputs.

        Args:
            labels (Optional[List[str]]): A list of label names, one per index in the
                output tensor. If no list is provided then numeric labels will be
                assigned according to the order of outputs.
        """
        self["modelType"] = "classification"
        self["labels"] = labels


class Regression(dict):
    def __init__(self):
        """Describe a regression output with a single value."""
        self["modelType"] = "regression"


class ObjectDetection(dict):
    def __init__(
        self,
        labels: List[str],
        last_layer: Union[
            Literal[
                "mobilenet-ssd", "fomo", "yolov5", "yolo5v5-drpai", "yolox", "yolov7"
            ],
            str,
        ],
        minimum_confidence: float,
    ):
        """Describe an object detection output with a specific format and labels.

        Args:
            labels (Optional[List[str]]): A list of label names, one per index in the
                output tensor. If no list is provided then numeric labels will be
                assigned according to the order of outputs.
            last_layer (Union[Literal, str]): The output type of the model, depending on
                the type of object detection model this is. Many common formats are
                supported. A full list can be found at https://docs.edgeimpulse.com/reference/edgeimpulse_apimodelsobject_detection_last_layer#edgeimpulse_apimodelsobject_detection_last_layer-module.
            minimum_confidence (float): The minimum confidence value, from 0 to 1.
                Detected objects with confidence scores below this value will be
                ignored. Set to 0 if you wish the model to return all detected objects.
        """
        self["modelType"] = "object-detection"
        self["labels"] = labels
        self["lastLayer"] = last_layer
        self["minimumConfidence"] = minimum_confidence
