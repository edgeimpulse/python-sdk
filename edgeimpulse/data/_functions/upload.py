# mypy: ignore-errors
# ruff: noqa: D100
from typing import Union, Optional, List
from urllib.parse import urljoin
from requests import Response, Session
from requests.adapters import HTTPAdapter

from concurrent.futures import ThreadPoolExecutor
import json
import random
import logging

import edgeimpulse as ei

from dataclasses import asdict

from edgeimpulse.data.sample_type import (
    Sample,
    DataAcquisition,
    SampleIngestionResponse,
    UploadSamplesResponse,
)
from edgeimpulse.exceptions import (
    MissingApiIngestionEndpointException,
    MissingApiKeyException,
)


def _upload_sample_with_progress(
    session: Session,
    sample: Sample,
    allow_duplicates: bool,
    api_key: str,
    endpoint: str,
    timeout_sec: Optional[float],
    progress_callback: Optional[callable] = None,
) -> Response:
    headers = {
        "x-api-key": api_key,
        "x-upload-source": "EDGE_IMPULSE_PYTHON_SDK",
    }

    # Construct headers
    if not allow_duplicates:
        headers["x-disallow-duplicates"] = "1"

    if sample.label:
        headers["x-label"] = str(sample.label)

    if sample.bounding_boxes:
        if isinstance(sample.bounding_boxes, list):
            sample.bounding_boxes = json.dumps(sample.bounding_boxes)
        headers["x-bounding-boxes"] = sample.bounding_boxes

    if sample.metadata:
        headers["x-metadata"] = json.dumps(sample.metadata)

    if sample.category is None:
        sample.category = "split"

    # Construct URL
    resource_path = f"/api/{sample.category}/files"
    url = urljoin(endpoint, resource_path)

    # Assign filename if not provided
    if sample.filename is None:
        sample.filename = ("%16x.json" % random.getrandbits(64),)

    # Fill in data
    if isinstance(sample.data, DataAcquisition):
        files = [
            (
                "data",
                (
                    sample.filename,
                    json.dumps(asdict(sample.data)),
                    "application/cbor",
                ),
            )
        ]
    else:
        files = [("data", (sample.filename, sample.data, "multipart/form-data"))]

    if sample.structured_labels:
        # append the structured labels to the file upload
        structured_labels = {
            "version": 1,
            "type": "structured-labels",
            "structuredLabels": {sample.filename: sample.structured_labels},
        }

        files.append(
            (
                "data",
                (
                    "structured_labels.labels",
                    json.dumps(structured_labels),
                    "application/json",
                ),
            )
        )

    # Make request
    response = session.post(url, headers=headers, files=files, timeout=timeout_sec)

    # Call progress callback
    if progress_callback:
        progress_callback()

    return (response, sample)


def _report_results(results: List[Response]) -> UploadSamplesResponse:
    """Convert http responses to a type with success and error samples."""
    successes = []
    fails = []

    # Sort results into successes and failures
    for result, sample in results:
        if result.status_code == 200:
            files = result.json().get("files", [])
            if len(files) == 0:
                fails.append(
                    SampleIngestionResponse(
                        sample,
                        {"success": False, "error": "No files listed in response"},
                    )
                )
            for file in files:
                if not file.get("success", False):
                    fails.append(SampleIngestionResponse(sample, file))
                else:
                    sample.sample_id = file.get("sampleId")
                    successes.append(SampleIngestionResponse(sample, file))
        else:
            try:
                err = result.json()
            except ValueError:
                err = {"error": result.text}
            fails.append(SampleIngestionResponse(sample, err))

    # Report results
    logging.info(f"Samples uploaded: {len(successes)}, failed: {len(fails)}.")
    if len(fails) > 0:
        [logging.info(item) for item in fails]

    # Create response object
    return UploadSamplesResponse(successes, fails)


def upload_samples(
    samples: Union[Sample, List[Sample]],
    allow_duplicates: Optional[bool] = False,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_workers: Optional[int] = None,
    show_progress: Optional[bool] = False,
    pool_maxsize: Optional[int] = 20,
    pool_connections: Optional[int] = 20,
) -> UploadSamplesResponse:
    """Upload one or more samples to an Edge Impulse project using the ingestion service.

    Each sample must be wrapped in a `Sample` object, which contains metadata about that sample.
    Give this function a single `Sample` or a List of `Sample` objects to upload to your
    project. The `data` field of the `Sample` must be a raw binary stream, such as a BufferedIOBase
    object (which you can create with the `open(..., "rb")` function).

    Args:
        samples (Union[Sample, List[Sample]]): One or more `Sample` objects that contain data
            for that sample along with associated metadata.
        allow_duplicates (Optional[bool]): Set to `True` to allow samples with the same data to be
            uploaded. If `False`, the ingestion service will perform a hash of the data and compare
            it to the hashes of the data already in the project. If a match is found, the service
            will reject the incoming sample (uploading for other samples will continue).
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or the env
            var `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Number of seconds to wait for an upload request to
            complete on the server. `None` is considered "infinite timeout" and will wait forever.
        max_workers (Optional[int]): The max number of workers to upload the samples. It should
            ideally be equal to the number of cores on your machine. If `None`, the number of
            workers will be automatically determined.
        show_progress (Optional[bool]): Show progress bar while uploading samples. Default is
            `False`.
        pool_maxsize (Optional[int]): The maximum number of connections to make in a single
            connection pool (for multithreaded uploads).
        pool_connections (Optional[int]): The maximum number of connections to cache for different
            hosts.

    Returns:
        UploadSamplesResponse: A response object that contains the results of the upload. The
            response object contains two tuples: the first tuple contains the samples that were
            successfully uploaded, and the second tuple contains the samples that failed to upload
            along with the error message.

    Examples:
        Upload samples

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        from edgeimpulse import data
        from edgeimpulse.data import Sample

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
        response = data.upload_samples(samples)
        print(response.successes)
        print(response.fails)
        ```
    """
    # Turn a single sample into a 1-element list
    if isinstance(samples, Sample):
        samples = [samples]

    api_key = api_key if api_key is not None else ei.API_KEY
    endpoint = ei.INGESTION_ENDPOINT
    samples = list(samples)  # TODO: support iterators?

    if not api_key:
        raise MissingApiKeyException()

    if not endpoint:
        raise MissingApiIngestionEndpointException()

    if show_progress:
        print()

    with Session() as session:
        adapter = HTTPAdapter(
            pool_connections=pool_connections, pool_maxsize=pool_maxsize
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_samples = len(samples)

            # Show progress bar if enabled
            if show_progress:
                print(f"Uploading {total_samples} samples...")

                def progress_callback():
                    nonlocal total_samples
                    total_samples -= 1
                    percent_complete = 100 - (total_samples / len(samples)) * 100
                    print(
                        f"Progress: {percent_complete:.2f}%",
                        end="\r",
                        flush=True,
                    )

            else:
                progress_callback = None

            results = list(
                executor.map(
                    lambda sample: _upload_sample_with_progress(
                        session,
                        sample,
                        allow_duplicates,
                        api_key,
                        endpoint,
                        timeout_sec,
                        progress_callback=progress_callback,
                    ),
                    samples,
                )
            )

    return _report_results(results)
