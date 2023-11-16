from dataclasses import dataclass
from io import BufferedIOBase
from typing import Optional, Sequence, Literal


@dataclass
class Sample:
    """
    Wrapper class for sample data, labels, and associated metadata. Sample data should be contained
    in a file or file-like object, for example, as the return from `open(..., "rb")`. The
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
            <https://docs.edgeimpulse.com/reference/image-dataset-annotation-formats>`_ for how to
            format bounding box dictionaries.
        metadata (Optional[dict]): Dictionary of optional metadata that you would like to include
            for your particular sample (example: `{"source": "microphone", "timestamp": "120"}`)
    """

    filename: str
    data: BufferedIOBase
    category: Optional[Literal["training", "testing", "anomaly", "split"]] = "split"
    label: Optional[str] = None
    bounding_boxes: Optional[Sequence[dict]] = None
    metadata: Optional[dict] = None
