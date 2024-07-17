"""Functions for building and downloading an impulse from Edge Impulse."""

import io
import logging
import os
import re

from typing import Optional

import edgeimpulse
import pydantic

from edgeimpulse.exceptions import (
    InvalidTargetException,
    InvalidEngineException,
    TimeoutException,
)

from edgeimpulse.util import (
    check_response_errors,
    configure_generic_client,
    default_project_id_for,
    get_project_deploy_targets,
    poll,
)

from edgeimpulse_api import (
    DeploymentApi,
    JobsApi,
)
from edgeimpulse_api.models.build_on_device_model_request import (
    BuildOnDeviceModelRequest,
)


def build(
    deploy_model_type: Optional[str] = None,
    engine: str = "tflite",
    deploy_target: str = "zip",
    output_directory: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> io.BytesIO:
    """Build and download an impulse from Edge Impulse.

    Build a model and download it from Edge Impulse. The model can be built for a specific target
    and engine. The model can be saved to a file if `output_directory` is provided and the file name
    will be derived from the deployment target.

    Args:
        deploy_model_type (str, optional): Use `int8` to receive an 8-bit quantized model
            `float32` for non-quantized. Defaults to None, in which case it will become `int8` if
            representative_data_for_quantization if provided and `float32` otherwise. For other
            values see `edgeimpulse.model.list_model_types()`.
        engine (str, optional): Inference engine. Either `tflite` (for TensorFlow Lite for
            Microcontrollers) or `tflite-eon` (for EON Compiler) to output a portable C++ library.
            For all engines, call `edgeimpulse.deploy.list_engines()`. Defaults to `tflite`.
        deploy_target (str, optional): Target to deploy to, defaulting to a portable C++ library
            suitable for most devices. See `edgeimpulse.model.list_deployment_targets()` for a list.
        output_directory (str, optional): Directory to write deployment artifact to. File name may
            vary depending on deployment type. Defaults to None in which case model will not be
            written to file.
        api_key (str, optional): The API key for an Edge Impulse project. This can also be set via
            the module-level variable `edgeimpulse.API_KEY`, or the env var `EI_API_KEY`.
            Defaults to None.
        timeout_sec (Optional[float], optional): Number of seconds to wait for profile job to
            complete on the server. `None` is considered "infinite timeout" and will wait forever.
            Defaults to None.

    Raises:
        InvalidTargetException: Raised if the target is invalid.
        InvalidEngineException: Raised if the engine is invalid.
        TimeoutException: Raised if the job times out.

    Returns:
        io.BytesIO: Built model.
    """
    # Configure API clients
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    deploy_api = DeploymentApi(client)
    jobs_api = JobsApi(client)

    # Get project ID
    project_id = default_project_id_for(client)

    # Check if the target is valid
    target_names = get_project_deploy_targets(client, project_id=project_id)
    if deploy_target not in target_names:
        raise InvalidTargetException(deploy_target, target_names)

    # Check if the engine is valid
    try:
        request = BuildOnDeviceModelRequest.from_dict(
            {"engine": engine, "modelType": deploy_model_type}
        )
    except pydantic.error_wrappers.ValidationError as e:
        if "Validation error for BuildOnDeviceModelRequest\nengine\n" in str(e):
            raise InvalidEngineException(e) from e
        raise e

    # Start deployment job
    try:
        response = jobs_api.build_on_device_model_job(
            project_id=project_id,
            type=deploy_target,
            build_on_device_model_request=request,
        )
        check_response_errors(response)
        job_id = response.id
    except Exception as e:
        logging.debug(f"Exception starting build job [{str(e)}]")
        raise e

    # Wait for deploy job to complete
    try:
        job_response = poll(
            jobs_client=jobs_api,
            project_id=project_id,
            job_id=job_id,
            timeout_sec=timeout_sec,
        )
    except TimeoutException as te:
        raise te
    except Exception as e:
        raise e
    logging.info(job_response)

    try:
        response = deploy_api.download_build(
            project_id=project_id,
            type=deploy_target,
            engine=engine,
            model_type=deploy_model_type,
            _preload_content=False,
        )
        logging.info(f"Deployment is {len(response.data)} bytes")
    except Exception as e:
        logging.debug(f"Exception downloading output [{str(e)}]")
        raise e

    # Write, as binary, to specified file.
    # Derive sensible name if none was provided
    if output_directory is not None:
        d = response.headers["Content-Disposition"]
        output_filename = re.findall(r"filename\*?=(.+)", d)[0].replace("utf-8''", "")
        output_path = os.path.join(output_directory, output_filename)
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            logging.info(f"Writing out to {output_path}")
            with open(output_path, "wb") as f:
                f.write(response.data)
        except Exception as e:
            logging.debug(f"Exception saving output to '{output_path}' [{str(e)}]")
            raise e

    return io.BytesIO(response.data)
