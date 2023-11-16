import logging
from typing import Optional, Tuple

import edgeimpulse
from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
)
from edgeimpulse_api import (
    RawDataApi,
)


def get_filename_by_id(
    sample_id: int,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[str]:
    """
    Given an ID for a sample in a project, return the filename associated with that
    sample.

    Note that while multiple samples can have the same filename, each sample has a
    unique sample ID that is provided by Studio when the sample is uploaded.

    Args:
        sample_id (int): Sample ID to look up
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API calls.

    Raises:
        e: Unhandled exception from api

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


def get_ids_by_filename(
    filename: str,
    category: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Tuple[int, ...]:
    """
    Given a filename for a sample in a project, return any sample IDs that match that
    filename.

    Note that filenames are given by the root of the filename when uploaded.
    For example, if you upload `my-image.01.png`, it will be stored in your project with
    a hash, such as `my-image.01.png.4f262n1b.json`. To find the ID(s) that match this
    sample, you must provide the argument `filename=my-image.01`. Notic the lack of
    extension and hash.

    Because of the possibility for multiple samples (i.e. different sample IDs) with the
    same filename, we recommend providing unique filenames for your samples.

    Args:
        filename (str): Root of the filename to look up (without extension or hash).
        category (Optional[str]): Category ("training", "testing", "anomaly") to look in
            for your sample. If no category is given, the function will look in all
            possible categories.
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API calls.

    Raises:
        e: Unhandled exception from api

    Returns:
        Tuple[int]: Tuple of IDs corresponding to the sample filename given. Empty if
            no samples matching the filename were found.
    """

    # Create API clients
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    raw_data_api = RawDataApi(client)

    # Get project ID associated with API key
    project_id = default_project_id_for(client)

    # Go through all categories if no category given
    if category is None:
        categories = edgeimpulse.util.DATA_CATEGORIES
    else:
        categories = [category]

    # Get IDs for a given filename
    try:
        resps = []
        for category in categories:
            resp = raw_data_api.list_samples(
                project_id=project_id,
                category=category,
                filename=filename,
                _request_timeout=timeout_sec,
            )
            resps.append(resp)
    except Exception as e:
        logging.debug(f"Exception trying to get sample IDs for {filename} [{str(e)}]")
        raise e

    # Construct list of IDs
    ids = []
    for resp in resps:
        if resp.samples is not None:
            for sample in resp.samples:
                ids.append(sample.id)

    return tuple(ids)
