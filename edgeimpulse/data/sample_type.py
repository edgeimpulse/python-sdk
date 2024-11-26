# ruff: noqa: D100, D101
from dataclasses import dataclass
from io import BufferedIOBase, StringIO
from typing import Optional, List, Literal, Union


@dataclass
class Sensor:
    """Represents a sensor in the Edge Impulse data acquisition format.

    Note:
        The units must comply with the SenML units list:
        https://www.iana.org/assignments/senml/senml.xhtml

    Attributes:
        name (str): Name of the sensor.
        units (Literal[
            "m", "kg", "g", "s", "A", "K", "cd", "mol", "Hz", "rad", "sr", "N", "Pa", "J", "W", "C", "V", "F", "Ohm",
            "S", "Wb", "T", "H", "Cel", "lm", "lx", "Bq", "Gy", "Sv", "kat", "m2", "m3", "l", "m/s", "m/s2", "m3/s", "l/s",
            "W/m2", "cd/m2", "bit", "bit/s", "lat", "lon", "pH", "dB", "dBW", "Bspl", "count", "/", "%", "%RH", "%EL",
            "EL", "1/s", "1/min", "beat/min", "beats", "S/m", "B", "VA", "VAs", "var", "vars", "J/m", "kg/m3", "deg", "NTU",
            "rgba"
        ]): Measurement units of the sensor as defined by https://www.iana.org/assignments/senml/senml.xhtml. Defaults to "m/s".
    """

    name: str
    units: Literal[
        "m",
        "kg",
        "g",
        "s",
        "A",
        "K",
        "cd",
        "mol",
        "Hz",
        "rad",
        "sr",
        "N",
        "Pa",
        "J",
        "W",
        "C",
        "V",
        "F",
        "Ohm",
        "S",
        "Wb",
        "T",
        "H",
        "Cel",
        "lm",
        "lx",
        "Bq",
        "Gy",
        "Sv",
        "kat",
        "m2",
        "m3",
        "l",
        "m/s",
        "m/s2",
        "m3/s",
        "l/s",
        "W/m2",
        "cd/m2",
        "bit",
        "bit/s",
        "lat",
        "lon",
        "pH",
        "dB",
        "dBW",
        "Bspl",
        "count",
        "/",
        "%",
        "%RH",
        "%EL",
        "EL",
        "1/s",
        "1/min",
        "beat/min",
        "beats",
        "S/m",
        "B",
        "VA",
        "VAs",
        "var",
        "vars",
        "J/m",
        "kg/m3",
        "deg",
        "NTU",
        "rgba",
    ] = "m/s"


@dataclass
class Payload:
    """Wrapper class for the sensor data.

    Information about the data acquisition format can be found here: https://docs.edgeimpulse.com/reference/data-acquisition-format.

    Attributes:
        device_type (str): - Device type, for example the exact model of the device.
            Should be the same for all similar devices. For example
            "DISCO-L475VG-IOT01A"
        sensors (List[Sensor]): List of Sensor objects representing the data acquisition
            sensors, such as `"accX"`, `"accY"`, `"accZ"` for a three-axis
            accelerometer where the units for each would be `"m/s2"`.
        values [List[int]]: List of lists containing float values representing the data
            from each of the sensors.
        interval_ms (Optional[int]): Interval in milliseconds between data samples.
            Default is 0.
        device_name (Optional[str]): Unique identifier for this device. Only set this
            when the device has a globally unique identifier (e.g. MAC address). If this
            field is set the device shows up on the 'Devices' page in the studio.

    Example:
        .. code-block:: python

            from edgeimpulse.data.sample_type import Payload, Sensor

            payload = Payload(
                device_type="DISCO-L475VG-IOT01A",
                sensors=[
                    Sensor(name="accX", units="m/s2"),
                    Sensor(name="accY", units="m/s2"),
                    Sensor(name="accZ", units="m/s2"),
                ],
                values=[
                    [-9.81, 0.03, 1.21],
                    [-9.83, 0.04, 1.27],
                    [-9.12, 0.03, 1.23],
                    [-9.14, 0.01, 1.25]
                ],
                interval_ms=10,
                device_name="ac:87:a3:0a:2d:1b"
            )
    """

    device_type: str
    sensors: List[Sensor]
    values: List[List[float]]
    interval_ms: Optional[int] = 0
    device_name: Optional[str] = None


@dataclass
class Protected:
    """Wrapper class for information about the signature format.

    More information can be found here: https://docs.edgeimpulse.com/reference/data-acquisition-format.

    Attributes:
        ver (str): Version of the signature format. Default is `"v1"`.
        alg (str): Algorithm used to generate the signature. Default is `"none"`.
        iat (Optional[int]): Date and time when the file was created in seconds since
            epoch. Only set this when the device creating the file has an accurate
            clock. Default is `None`.

    Example:
        .. code-block:: python

            from edgeimpulse.data.sample_type import Protected

            protected = Protected(
                ver="v1",
                alg="none",
                iat=1609459200
            )
    """

    ver: str = "v1"
    alg: Literal["HS256", "none"] = "none"
    iat: Optional[int] = None


@dataclass
class DataAcquisition:
    """Wrapper class for the Edge Impulse data acquisition format.

    See here for more information: https://docs.edgeimpulse.com/reference/data-acquisition-format.

    Attributes:
        protected (Protected): Information about the signature format.
        payload (Payload): Sensor data.
        signature (Optional[str]): Cryptographic signature of the data. Default is
            `None`.
    """

    protected: Protected
    payload: Payload
    signature: Optional[str] = None


@dataclass
class Sample:
    """Wrapper class for sample data, labels, and associated metadata.

    Sample data should be contained in a file or file-like object, for example, as the return from `open(..., "rb")`. The
    `upload_samples()` function expects Sample objects as input.

    Attributes:
        filename (str): Name to give the sample when stored in the Edge Impulse project
        data (BufferedIOBase): IO stream of data to be read during the upload process. This can be
            a BytesIO object, such as the return from `open(..., "rb")`.
        category (Optional[Literal["training", "testing", "anomaly", "split"]]): Which dataset to
            store your sample. The default, "split," lets the Edge Impulse server randomly assign
            the location of your sample based on the project's split ratio (default 80% training and
            20% test).
        label (Optional[str]): The label to assign to your sample for classification and regression
            tasks.
        bounding_boxes (Optional[dict]): Array of dictionary objects that define the bounding boxes
            for a given sample (object detection projects only). See `our image annotation guide
            <https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/uploader#understanding-image-dataset-annotation-formats>`_ for how to
            format bounding box dictionaries.
        metadata (Optional[dict]): Dictionary of optional metadata that you would like to include
            for your particular sample (example: `{"source": "microphone", "timestamp": "120"}`)
        sample_id (Optional[int]): Unique ID of the sample. This is automatically assigned by the
            Edge Impulse server when the sample is uploaded. You can use this ID to retrieve the
            sample later. This value is ignored when uploading samples and should not be set by the
            user.
        structured_labels (Optional[List[dict]]): Array of dictionary objects that define the labels
            in this sample at various intervals. See `the multi label guide
            <https://edge-impulse.gitbook.io/docs/edge-impulse-studio/data-acquisition/multi-label>`_ to
            read more. Example: `[{"label": "noise","startIndex": 0,"endIndex": 5000},
            {"label": "water","startIndex": 5000,"endIndex": 10000}]`

    """

    data: Union[BufferedIOBase, StringIO, str]
    filename: Optional[str] = None
    category: Optional[Literal["training", "testing", "anomaly", "split"]] = "split"
    label: Optional[str] = None
    bounding_boxes: Optional[List[dict]] = None
    metadata: Optional[dict] = None
    sample_id: Optional[int] = None
    structured_labels: Optional[List[dict]] = None

    def __str__(self) -> str:
        """Sample representation."""
        return (
            f"Sample(filename={self.filename!r}, "
            f"category={self.category!r}, "
            f"label={self.label!r}, "
            f"metadata={self.metadata!r}, "
            f"sample_id={self.sample_id!r})"
        )


class SampleIngestionResponse:
    def __init__(
        self,
        sample: Sample,
        response: dict,
    ):
        """Wrapper for the response from the Edge Impulse ingestion service when uploading a sample along with the sample that was uploaded.

        Args:
            sample (Sample): The sample that was uploaded.
            response (dict): The response from the server.
        """  # noqa: D401
        self.sample = sample
        self.success = False
        self.error = None
        self.sample_id = None
        self.project_id = None
        self.filename = None

        # Assign attributes if they exist in the response
        if "success" in response:
            self.success = response["success"]
        if "error" in response:
            self.error = response["error"]
        if "sampleId" in response:
            self.sample_id = response["sampleId"]
        if "projectId" in response:
            self.project_id = response["projectId"]
        if "fileName" in response:
            self.filename = response["fileName"]

    def __repr__(self) -> str:
        """Return a string providing an overview of the response.

        Returns:
            str: Response from the server.
        """
        msg = f"SampleIngestionResponse(success={self.success}, error={self.error}, "
        msg += f"sampleId={self.sample_id}, projectId={self.project_id}, "
        msg += f"fileName={self.filename}, sample={self.sample})"

        return msg


class UploadSamplesResponse:
    def __init__(
        self,
        successes: List[SampleIngestionResponse],
        fails: List[SampleIngestionResponse],
    ):
        """Response from the Edge Impulse server when uploading multiple samples.

        Args:
            success (bool): `True` if all samples were uploaded successfully.
            successes (List[SampleIngestionResponse]): List of
                SampleIngestionResponse objects for each sample that were successfully
                uploaded.
            fails (List[SampleIngestionResponse]): List of SampleIngestionResponse
                objects for each sample that failed to upload.
        """
        self.successes = successes
        self.fails = fails
        self.success = len(fails) == 0

    def __repr__(self) -> str:
        """Return a string providing an overview of the number of successes and fails from the responses.

        Returns:
            str: Successful upload along with number of successes and fails.
        """
        msg = "UploadSamplesResponse\r\n"
        msg += f"  success: {self.success}\r\n"
        msg += f"  number of successes: {len(self.successes)}\r\n"
        msg += f"  number of fails: {len(self.fails)}"

        return msg

    def extend(
        self,
        successes: List[SampleIngestionResponse],
        fails: List[SampleIngestionResponse],
    ):
        """Add new responses to the existing responses.

        Args:
            successes (List[SampleIngestionResponse]): List of
                SampleIngestionResponse objects for each sample that were successfully
                uploaded.
            fails (List[SampleIngestionResponse]): List of SampleIngestionResponse
                objects for each sample that failed to upload.
        """
        self.successes.extend(successes)
        self.fails.extend(fails)
        self.success = len(self.fails) == 0


@dataclass
class SampleInfo:
    """Wrapper for the response from the Edge Impulse ingestion service when retrieving sample information.

    Attributes:
        sample_id (Optional[int]): The sample ID.
        filename (Optional[str]): The filename of the sample.
        category (Optional[str]): The category of the sample.
        label (Optional[str]): The label of the sample.
    """

    sample_id: Optional[int] = None
    filename: Optional[str] = None
    category: Optional[str] = None
    label: Optional[str] = None
    category = None
    label = None
