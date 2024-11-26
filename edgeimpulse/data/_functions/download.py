# mypy: ignore-errors
# ruff: noqa: D100
import json
import logging
import concurrent.futures
import os
from io import BytesIO
from typing import Optional, Sequence, Union, List, Generator
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError

import edgeimpulse as ei
from edgeimpulse.exceptions import (
    UnsupportedSampleType,
)
from edgeimpulse.data.sample_type import (
    Sample,
    DataAcquisition,
    Sensor,
    Protected,
    Payload,
)
from edgeimpulse.util import (
    get_user_agent,
    configure_generic_client,
    default_project_id_for,
)


def _download_sample_as_image(
    session: Session,
    project_id: int,
    sample_id: int,
    api_key: str,
    endpoint: str,
    timeout_sec: Optional[float] = None,
) -> Optional[BytesIO]:
    """Download a sample by ID from an Edge Impulse project as an image."""
    data = None

    # Workaround: Use raw request for now
    resource_path = f"/v1/api/{project_id}/raw-data/{sample_id}/image"
    url = urljoin(endpoint, resource_path)
    headers = {
        "accept": "application/json",
        "x-api-key": api_key,
    }
    resp_raw = session.get(url, headers=headers, timeout=timeout_sec)
    if resp_raw.status_code == 200:
        data = BytesIO(resp_raw.content)
    else:
        raise HTTPError(
            f"HTTP error occurred: {resp_raw.status_code} - {resp_raw.reason}"
        )

    return data


def _download_sample_as_video(
    session: Session,
    project_id: int,
    sample_id: int,
    api_key: str,
    endpoint: str,
    timeout_sec: Optional[float] = None,
) -> Optional[BytesIO]:
    """Download a sample by ID from an Edge Impulse project as a video."""
    data = None

    # Workaround: Use raw request for now
    resource_path = f"/v1/api/{project_id}/raw-data/{sample_id}/video"
    url = urljoin(endpoint, resource_path)
    headers = {
        "accept": "application/json",
        "x-api-key": api_key,
    }
    resp_raw = session.get(url, headers=headers, timeout=timeout_sec)
    if resp_raw.status_code == 200:
        data = BytesIO(resp_raw.content)
    else:
        raise HTTPError(
            f"HTTP error occurred: {resp_raw.status_code} - {resp_raw.reason}"
        )

    return data


def _download_sample_as_audio(
    session: Session,
    project_id: int,
    sample_id: int,
    api_key: str,
    endpoint: str,
    timeout_sec: Optional[float] = None,
) -> Optional[BytesIO]:
    """Download a sample by ID from an Edge Impulse project as a .WAV file."""
    data = None

    # Workaround: Use raw request for now
    resource_path = f"/v1/api/{project_id}/raw-data/{sample_id}/wav?axisIx=0"
    url = urljoin(endpoint, resource_path)
    headers = {
        "accept": "application/json",
        "x-api-key": api_key,
    }
    resp_raw = session.get(url, headers=headers, timeout=timeout_sec)
    if resp_raw.status_code == 200:
        data = BytesIO(resp_raw.content)
    else:
        raise HTTPError(
            f"HTTP error occurred: {resp_raw.status_code} - {resp_raw.reason}"
        )

    return data


def _convert_timeseries_to_json(
    resp: dict,
) -> BytesIO:
    """Parse time series response from Edge Impulse API and convert to CBOR buffer."""
    # Parse response
    data = DataAcquisition(
        protected=Protected(),
        payload=Payload(
            device_type=resp["payload"]["device_type"],
            sensors=[
                Sensor(
                    name=sensor["name"],
                    units=sensor["units"],
                )
                for sensor in resp["sample"]["sensors"]
            ],
            values=resp["payload"]["values"],
            interval_ms=resp["sample"]["intervalMs"],
            device_name=resp["payload"]["device_name"],
        ),
    )

    # Convert to CBOR
    data = json.dumps(asdict(data))

    return data


def _download_sample_with_progress(
    session: Session,
    project_id: int,
    sample_id: int,
    api_key: str,
    endpoint: str,
    timeout_sec: Optional[float] = None,
    progress_callback: Optional[callable] = None,
) -> Optional[Sample]:
    """Download a sample by ID from an Edge Impulse project with progress info."""
    # Define headers
    headers = {
        "x-api-key": api_key,
        "accept": "application/json",
        "User-Agent": get_user_agent(add_platform_info=True),
    }

    # Define endpoint
    if not endpoint.endswith("/"):
        endpoint += "/"
    resource_path = f"api/{project_id}/raw-data/{sample_id}"
    url = urljoin(endpoint, resource_path)

    # Make request
    resp = session.get(url, headers=headers, timeout=timeout_sec)
    if resp.status_code != 200:
        raise HTTPError(f"HTTP error occurred: {resp.status_code} - {resp.reason}")

    # Convert JSON response to dictionary
    resp = resp.json()
    if isinstance(resp, str):
        resp = json.loads(resp)

    # Check for unsuccessful response
    if resp["success"] is False:
        logging.info(f"Could not get sample with ID {sample_id}")
        return None

    # Extract the filename
    filename = resp["sample"]["coldstorageFilename"]
    filename = os.path.basename(filename)
    filename = ".".join(filename.split(".")[:-3])

    # Convert JSON and CBOR filename to CSV
    if os.path.splitext(filename)[1] in [".json", ".cbor"]:
        filename = os.path.splitext(filename)[0] + ".csv"

    # Determine file type
    filetype = resp["payload"]["sensors"][0]["name"]

    # Treat "wav" as its own type
    if filetype == "audio" and resp["payload"]["sensors"][0]["units"] == "wav":
        filetype = "wav"

    # Download raw image
    if filetype == "image":
        data = _download_sample_as_image(
            session=session,
            project_id=project_id,
            sample_id=sample_id,
            api_key=api_key,
            endpoint=endpoint,
            timeout_sec=timeout_sec,
        )

    # Download raw video
    elif filetype == "video":
        data = _download_sample_as_video(
            session=session,
            project_id=project_id,
            sample_id=sample_id,
            api_key=api_key,
            endpoint=endpoint,
            timeout_sec=timeout_sec,
        )

    # Download raw wav
    elif filetype == "wav":
        data = _download_sample_as_audio(
            session=session,
            project_id=project_id,
            sample_id=sample_id,
            api_key=api_key,
            endpoint=endpoint,
            timeout_sec=timeout_sec,
        )

    # Extract raw data time series data from the response
    elif (
        "originalIntervalMs" in resp["sample"]
        and resp["sample"]["originalIntervalMs"] > 0
    ):
        data = _convert_timeseries_to_json(resp)

        # Change the filename extension to JSON
        filename = os.path.splitext(filename)[0] + ".json"

    # Unsupported filetype
    else:
        raise UnsupportedSampleType(resp)

    # Return None if no data
    if data is None:
        return None

    # Wrap sample
    sample = Sample(
        data=data,
        filename=filename,
        sample_id=sample_id,
        label=resp["sample"]["label"],
        category=resp["sample"]["category"],
        bounding_boxes=resp["sample"]["boundingBoxes"],
        metadata=resp["sample"].get("metadata", None),
        structured_labels=resp["sample"].get("structuredLabels", None),
    )

    if progress_callback:
        progress_callback()

    return sample


def stream_samples_by_ids(
    sample_ids: Union[int, Sequence[int]],
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_workers: Optional[int] = None,
    show_progress: Optional[bool] = False,
    pool_maxsize: Optional[int] = 20,
    pool_connections: Optional[int] = 20,
) -> Generator[Sample, None, None]:
    """Download samples by their associated IDs from an Edge Impulse project.

    Args:
        sample_ids (Union[int, Sequence[int]]): IDs of the samples to download
        api_key (Optional[str]): The API key for an Edge Impulse project.
        timeout_sec (float, optional): Number of seconds to wait for profile
            job to complete on the server. `None` is considered "infinite timeout".
        max_workers (int, optional): The maximum number of subprocesses to use.
        show_progress: Show progress bar while uploading samples.
        pool_maxsize: (int, optional) Maximum size of the upload pool.
        pool_connections: (int, optional) Maximum size of the pool connections.

    Yields:
        Sample: A Sample object with data and metadata as downloaded from
        the Edge Impulse project. Will yield `None` if a sample with the
        matching ID is not found.
    """
    if isinstance(sample_ids, int):
        sample_ids = [sample_ids]

    sample_ids = list(sample_ids)
    if not all(isinstance(sample_id, int) for sample_id in sample_ids):
        raise TypeError("All sample IDs must be integers")

    api_key = api_key if api_key is not None else ei.API_KEY
    endpoint = ei.API_ENDPOINT

    client = configure_generic_client(key=api_key, host=endpoint)
    project_id = default_project_id_for(client)

    if show_progress:
        print()

    with Session() as session:
        adapter = HTTPAdapter(
            pool_connections=pool_connections, pool_maxsize=pool_maxsize
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_samples = len(sample_ids)

            if show_progress:
                print(f"Looking for {total_samples} samples to download...")

                def progress_callback():
                    nonlocal total_samples
                    total_samples -= 1
                    percent_complete = 100 - (total_samples / len(sample_ids)) * 100
                    print(f"Progress: {percent_complete:.2f}%")

            else:
                progress_callback = None

            futures = {
                executor.submit(
                    _download_sample_with_progress,
                    session,
                    project_id,
                    sample_id,
                    api_key,
                    endpoint,
                    timeout_sec,
                    progress_callback,
                ): sample_id
                for sample_id in sample_ids
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    yield result
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise

    logging.info(
        f"Downloaded {total_samples} samples of the requested {len(sample_ids)} IDs"
    )


def download_samples_by_ids(
    sample_ids: Union[int, List[int]],
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_workers: Optional[int] = None,
    show_progress: Optional[bool] = False,
    pool_maxsize: Optional[int] = 20,
    pool_connections: Optional[int] = 20,
) -> List[Sample]:
    """Download samples by their associated IDs from an Edge Impulse project.

    Downloaded sample data is returned as a `DownloadSample` object, which contains the raw data in
    a BytesIO object along with associated metadata.

    **Important!** All time series data is returned as a JSON file (in BytesIO format)
    with a timestamp column. This includes files originally uploaded as CSV, JSON, and
    CBOR. Edge Impulse Studio removes the timestamp column from any uploaded CSV
    files and computes an estimated sample rate. The timestamps are computed based on
    the sample rate, will always start at 0, and will be in milliseconds. These
    timestamps may not be the same as the original timestamps in the uploaded file.

    Args:
        sample_ids (Union[int, List[int]]): IDs of the samples to download
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the env var `EI_API_KEY`.
        timeout_sec (float, optional): Number of seconds to wait for profile
            job to complete on the server. `None` is considered "infinite timeout" and
            will wait forever.
        max_workers (int, optional): The maximum number of subprocesses to use when
            making concurrent requests. If `None`, the number of workers will be set to
            the number of processors on the machine multiplied by 5.
        show_progress: Show progress bar while uploading samples.
        pool_maxsize: (int, optional) Maximum size of the upload pool. Defaults to 20.
        pool_connections: (int, optional) Maximum size of the pool connections. Defaults to 20.

    Returns:
        List[Sample]: List of Sample objects with data and metadata as downloaded from
            the Edge Impulse project. Will be an empty list `[]` if no samples
            with the matching IDs are found.

    Example:
        Download a sample by id

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY

        sample = ei.data.download_samples_by_ids(12345)
        print(sample)
        ```
    """
    # Turn single or multiple IDs into list
    if isinstance(sample_ids, int):
        sample_ids = [sample_ids]
    sample_ids = list(sample_ids)

    # Type check to ensure all IDs are integers
    if not all(isinstance(sample_id, int) for sample_id in sample_ids):
        raise TypeError("All sample IDs must be integers")

    # Define endpoint and API key
    api_key = api_key if api_key is not None else ei.API_KEY
    endpoint = ei.API_ENDPOINT

    # Get project ID associated with API key
    client = configure_generic_client(
        key=api_key,
        host=endpoint,
    )
    project_id = default_project_id_for(client)

    # Download multiple samples concurrently
    if show_progress:
        print()

    # Download samples. Keep the session open so we can reuse the connection to
    # make multiple requests, which is faster than opening a new connection for
    # each request.
    with Session() as session:
        adapter = HTTPAdapter(
            pool_connections=pool_connections, pool_maxsize=pool_maxsize
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_samples = len(sample_ids)

            # Define callback to update progress
            if show_progress:
                print(f"Looking for {total_samples} samples to download...")

                def progress_callback():
                    nonlocal total_samples
                    total_samples -= 1
                    percent_complete = 100 - (total_samples / len(sample_ids)) * 100
                    print(
                        f"Progress: {percent_complete:.2f}%",
                        end="\r",
                        flush=True,
                    )

            else:
                progress_callback = None

            # Map upload functions to executor
            samples = list(
                executor.map(
                    lambda sample_id: _download_sample_with_progress(
                        session=session,
                        project_id=project_id,
                        sample_id=sample_id,
                        api_key=api_key,
                        endpoint=endpoint,
                        timeout_sec=timeout_sec,
                        progress_callback=progress_callback,
                    ),
                    sample_ids,
                )
            )

    # Remove any None values
    samples = [sample for sample in samples if sample is not None]

    # Log how many samples were actually found and downloaded
    logging.info(
        f"Downloaded {len(samples)} samples of the requested {len(sample_ids)} IDs"
    )

    return samples
