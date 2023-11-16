import json
from typing import Union, Optional, Tuple, Sequence
from urllib.parse import urljoin

from requests import post, Response

import edgeimpulse as ei
from edgeimpulse.data.sample_type import (
    Sample,
)


def upload_samples(
    samples: Union[Sample, Sequence[Sample]],
    allow_duplicates: Optional[bool] = False,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Tuple[Response]:
    """
    Uploads one or more samples to an Edge Impulse project using the ingestion service.

    Each sample must be wrapped in a `Sample` object, which contains metadata about that sample.
    Give this function a single `Sample` or a sequence of `Sample` objects to upload to your
    project. The `data` field of the `Sample` must be a raw binary stream, such as a BufferedIOBase
    object (which you can create with the `open(..., "rb")` function).

    Args:
        samples (Union[Sample, Sequence[Sample]]): One or more `Sample` objects that contain data
            for that sample along with associated metadata.
        allow_duplicates (Optional[bool]): Set to `True` to allow samples with the same data to be
            uploaded. If `False`, the ingestion service will perform a hash of the data and compare
            it to the hashes of the data already in the project. If a match is found, the service
            will reject the incoming sample (uploading for other samples will continue).
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or the env
            var `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Number of seconds to wait for profile job to
            complete on the server. `None` is considered "infinite timeout" and will wait forever.

    Returns:
        Tuple[Response]: Sequence of requests. Response objects that contain information about the
            upload status of your Samples.

    Examples:

        .. code-block:: python

            # Create a dataset (with a single Sample)
            samples = (
                Sample(
                    filename="wave.01.csv",
                    data=open("path/to/wave.01.csv", "rb"),
                    category="split",
                    label="wave",
                ),
            )

            # Upload samples and print responses
            resps = ei.data.upload_samples(samples)
            for resp in resps:
                print(resp)
    """

    # Turn a single sample into a 1-element list
    if isinstance(samples, Sample):
        samples = [samples]

    # Define endpoint and API key
    api_key = api_key if api_key is not None else ei.API_KEY
    endpoint = ei.INGESTION_ENDPOINT

    # Send one request per sample
    # TODO: Update this to send multiple samples per request once Studio supports it
    resps = []
    for sample in samples:
        # Configure headers
        headers = {
            "x-api-key": api_key,
            "x-upload-source": "EDGE_IMPULSE_PYTHON_SDK",
        }
        if not allow_duplicates:
            headers["x-disallow-duplicates"] = "1"
        if sample.label:
            headers["x-label"] = sample.label
        if sample.bounding_boxes:
            headers["x-bounding-boxes"] = sample.bounding_boxes
        if sample.metadata:
            headers["x-metadata"] = json.dumps(sample.metadata)

        # Construct full URL
        resource_path = f"/api/{sample.category}/files"
        url = urljoin(endpoint, resource_path)

        # Construct required tuple for uploading
        # TODO: construct multiple tuples when Studio allows it and
        # add per-file header information in a 4-tuple (instead of 3)
        files = [("data", (sample.filename, sample.data, "multipart/form-data"))]

        # Upload data
        resp = post(
            url=url,
            headers=headers,
            files=files,
            timeout=timeout_sec,
        )

        resps.append(resp)

    return tuple(resps)
