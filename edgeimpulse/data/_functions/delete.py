import logging
from typing import Optional

import edgeimpulse
from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
)
from edgeimpulse_api import (
    RawDataApi,
    GenericApiResponse,
)


def delete_all_samples(
    category: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[GenericApiResponse]:
    """
    Deletes all samples in a given category. If category is set to `None`, all samples
    in the project are deleted.

    Args:
        category (Optional[str]): Category ("training", "testing", "anomaly") from which
            the samples should be deleted. Set to 'None' to delete all samples from all
            categories.
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API calls.

    Raises:
        e: Unhandled exception from api

    Returns:
        Optional[GenericApiResponse]: API response
    """

    # Create API clients
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    raw_data_api = RawDataApi(client)

    # Get project ID associated with API key
    project_id = default_project_id_for(client)

    # Delete sample
    try:
        if category is None:
            resp = raw_data_api.delete_all_samples(
                project_id=project_id,
                _request_timeout=timeout_sec,
            )
        else:
            resp = raw_data_api.delete_all_samples_by_category(
                project_id=project_id,
                category=category,
                _request_timeout=timeout_sec,
            )
    except Exception as e:
        logging.debug(f"Exception trying to delete samples [{str(e)}]")
        raise e

    return resp


def delete_sample_by_id(
    sample_id: int,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[GenericApiResponse]:
    """
    Delete a particular sample from a project given the sample ID.

    Args:
        sample_id (id): ID of the sample to delete
        api_key (Optional[str]): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or
            the environment variable `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Optional timeout (in seconds) for API calls.

    Raises:
        e: Unhandled exception from api

    Returns:
        Optional[GenericApiResponse]: API response, None if no sample is found

    Examples:

        .. code-block:: python

            # Example of filename that has been uploaded to Studio
            filename = "my-image.01.png"

            # Remove extension on the filename when querying the dataset in Studio
            filename_no_ext = os.path.splitext(filename)[0]

            # Get list of IDs that match the given sample filename
            ids = ei.data.get_ids_by_filename(filename_no_ext)

            # Delete the IDs
            for id_num in ids:
                resp = ei.data.delete_sample_by_id(id_num)
                if resp is None:
                    logging.warning(f"Could not delete sample {filename_no_ext}")
    """

    # Create API clients
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    raw_data_api = RawDataApi(client)

    # Get project ID associated with API key
    project_id = default_project_id_for(client)

    # Delete sample
    try:
        resp = raw_data_api.delete_sample(
            project_id=project_id,
            sample_id=sample_id,
            _request_timeout=timeout_sec,
        )
    except Exception as e:
        if str(e).startswith("No sample found with ID"):
            logging.info(f"No sample found with ID {sample_id}")
            return None
        else:
            logging.debug(f"Exception trying to delete sample {sample_id} [{str(e)}]")
            raise e

    logging.info(f"Deleted sample {sample_id}")

    return resp
