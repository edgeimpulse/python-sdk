# ruff: noqa: D100, D101
from typing import Literal


class ImageInput(dict):
    def __init__(self, scaling_range: Literal["0..1", "0..255", "torch"] = "0..1"):
        """Describe an image input, and specifies how it should be processed.

        Args:
            scaling_range (Literal['0..1', '0..255', 'torch']): Describes any scaling or
                normalization that is applied to images. If no value is set then "0..1" is used.
                "0..1" gives you non-normalized pixels between 0 and 1. "0..255" gives you
                non-normalized pixels between 0 and 255. "torch" first scales pixels between 0 and
                1, then applies normalization using the ImageNet dataset as a reference (same as
                `torchvision.transforms.Normalize()`).
        """
        self["inputType"] = "image"
        self["inputScaling"] = scaling_range


class AudioInput(dict):
    def __init__(self, frequency_hz: float):
        """Describe an audio input, and specifies how it should be processed.

        Args:
            frequency_hz (float): The frequency of the audio signal in Hz (samples per second).
        """
        self["inputType"] = "audio"
        self["frequencyHz"] = frequency_hz


class TimeSeriesInput(dict):
    def __init__(self, frequency_hz: float, windowlength_ms: int):
        """Describe a time series input, and specifies how it should be processed.

        A stream of time series data is windowed into chunks of `windowlength_ms` according to the
        signal's frequency (`frequency_hz`). The window length represents how much data
        is fed into the model per inference.

        Args:
            frequency_hz (float): The frequency of the signal in Hz (samples per second).
            windowlength_ms (int): The length of the window of data that is fed into the model each
            inference.
        """
        self["inputType"] = "time-series"
        self["frequencyHz"] = frequency_hz
        self["windowLengthMs"] = windowlength_ms


class OtherInput(dict):
    def __init__(self):
        """Describe an input that is passed into the model without any changes."""
        self["inputType"] = "other"
