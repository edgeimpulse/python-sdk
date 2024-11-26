# mypy: ignore-errors
"""Use this module to do various tasks within Edge Impulse SDK."""
import json
import logging
import math
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed


import edgeimpulse
from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
)
from edgeimpulse.data.sample_type import (
    SampleInfo,
)

from edgeimpulse_api import (
    RawDataApi,
)


def get_filename_by_id(
    sample_id: int,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[str]:
    """Given an ID for a sample in a project, return the filename associated with that sample.

    Note that while multiple samples can have the same filename, each sample has a
    unique sample ID that is provided by Studio when the sample is uploaded.

    Args:
        sample_id (int): Sample ID to look up.
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API calls.

    Raises:
        e: Unhandled exception from api.

    Returns:
        Optional[str]: Filename (string) if sample is found. None if no sample is found
            matching the ID given.
    """
    # Create API clients
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    raw_data_api = RawDataApi(client)

    # Get project ID associated with API key
    project_id = default_project_id_for(client)

    # Get filename for a given ID
    try:
        resp = raw_data_api.get_sample(
            project_id=project_id,
            sample_id=sample_id,
            _request_timeout=timeout_sec,
        )
        if resp.sample is not None:
            return resp.sample.filename
    except Exception as e:
        if str(e).startswith("No sample found with ID"):
            logging.info(f"No sample found with ID {sample_id}")
            return None
        else:
            logging.debug(
                f"Exception trying to get filename for sample {sample_id} [{str(e)}]"
            )
            raise e

    return None


def _list_samples(
    raw_data: RawDataApi,
    project_id: str,
    category: str,
    labels: List[str],
    filename: str,
    offset: int,
    samples_per_thread: int,
    chunk_size: int = 1000,
    timeout_sec=None,
):
    """Make API calls to get sample info from a project."""
    # Determine how many times to make API call
    num_chunks = int(math.ceil(samples_per_thread / chunk_size))

    # Make API calls to list sample information
    responses = []
    for i in range(num_chunks):
        # Determine offset and limit for this chunk
        chunk_offset = i * chunk_size
        limit = min(samples_per_thread - chunk_offset, chunk_size)

        # Make API call
        try:
            resp = raw_data.list_samples(
                project_id=project_id,
                category=category,
                labels=json.dumps(labels),
                filename=filename,
                offset=offset + chunk_offset,
                limit=limit,
                _request_timeout=timeout_sec,
            )
        except Exception as e:
            logging.debug(f"Exception trying to get sample info [{str(e)}]")
            raise e

        # Parse and combine responses
        for sample in resp.samples:
            responses.append(
                SampleInfo(
                    sample_id=sample.id,
                    filename=sample.filename,
                    category=sample.category,
                    label=sample.label,
                )
            )

    return responses


def get_sample_ids(
    filename: Optional[str] = None,
    category: Optional[str] = None,
    labels: Optional[str] = None,
    api_key: Optional[str] = None,
    num_workers: Optional[int] = 4,
    timeout_sec: Optional[float] = None,
) -> List[SampleInfo]:
    """Get the sample IDs and filenames for all samples in a project, filtered by category, labels, or filename.

    Note that filenames are given by the root of the filename when uploaded.
    For example, if you upload `my-image.01.png`, it will be stored in your project with
    a hash such as `my-image.01.png.4f262n1b.json`. To find the ID(s) that match this
    sample, you must provide the argument `filename=my-image.01`. Notice the lack of
    extension and hash.

    Because of the potential for multiple samples (i.e., different sample IDs) with the
    same filename, we recommend providing unique filenames for your samples when
    uploading.

    Args:
        filename (Optional[str]): Filename of the sample(s) (without extension or hash)
            to look up. Note that multiple samples can have the same filename. If no
            filename is given, the function will look for samples with any filename.
        category (Optional[str]): Category ("training", "testing", "anomaly") to look in
            for your sample. If no category is given, the function will look in all
            possible categories.
        labels (Optional[str]): Label to look for in your sample. If no label is given,
            the function will look for samples with any label.
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        num_workers (Optional[int]): Number of threads to use to make API calls.
            Defaults to 4.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API
            calls.

    Raises:
        e: Unhandled exception from API.

    Returns:
        List[SampleInfo]: List of `SampleInfo` objects containing the sample ID,
            filename, category, and label for each sample matching the criteria given.
    """
    # Recursively get info from all categories if no category is given
    if category == "all" or category is None:
        resp_samples = []
        for category in edgeimpulse.util.DATA_CATEGORIES:
            resp_samples.extend(
                get_sample_ids(
                    category=category,
                    labels=labels,
                    filename=filename,
                    num_workers=num_workers,
                )
            )

        return resp_samples

    # Check to make sure category is in the allowed set
    if category not in edgeimpulse.util.DATA_CATEGORIES:
        raise ValueError(
            "Invalid category. Allowable categories: "
            f"{edgeimpulse.util.DATA_CATEGORIES} "
            "and None to search in all categories."
        )

    # Configure API client (TODO: make sure user can pass in api_key)
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    # Get project ID
    project_id = default_project_id_for(client)

    # Configure specific API client
    raw_data = RawDataApi(client)

    # Make labels into a list
    if isinstance(labels, str):
        labels = [labels]

    # Get the number of samples with an API call
    resp = raw_data.count_samples(
        project_id=project_id,
        category=category,
        labels=json.dumps(labels),
        _request_timeout=timeout_sec,
    )
    total_samples = resp.count

    # Compute start and stop index for each thread
    samples_per_thread = int(math.ceil(total_samples / num_workers))
    start_indexes = []
    for i in range(num_workers):
        start_indexes.append(i * samples_per_thread)

    # Create a ThreadPoolExecutor
    resp_samples = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the pool
        futures = [
            executor.submit(
                _list_samples,
                raw_data=raw_data,
                project_id=project_id,
                category=category,
                labels=labels,
                filename=filename,
                offset=start_indexes[i],
                samples_per_thread=samples_per_thread,
                timeout_sec=timeout_sec,
            )
            for i in range(num_workers)
        ]

        # Collect results as they are completed
        for future in as_completed(futures):
            resp_samples.extend(future.result())

    return resp_samples
