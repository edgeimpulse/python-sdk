# mypy: ignore-errors
import logging
import json
from pathlib import Path
from typing import Union, Optional, Any, List
import tempfile

import edgeimpulse
from edgeimpulse.exceptions import (
    InvalidDeviceException,
    TimeoutException,
)

from edgeimpulse.util import (
    configure_generic_client,
    poll,
    default_project_id_for,
    get_profile_devices,
    upload_pretrained_model_and_data,
    check_response_errors,
)
from edgeimpulse_api import (
    JobsApi,
    LearnApi,
    GetPretrainedModelResponse,
)


class ProfileResponse(GetPretrainedModelResponse):
    def summary(self) -> None:
        """Return a summary of the profiling results."""
        output = []
        if self.specific_device_selected and self.model and self.model.profile_info:
            if self.model.profile_info.float32:
                output.append("Target results for float32:")
                output.append("===========================")
                output.append(
                    json.dumps(
                        self.model.profile_info.float32.to_dict(),
                        indent=4,
                    )
                )
                output.append("\n")
            if self.model.profile_info.int8:
                output.append("Target results for int8:")
                output.append("========================")
                output.append(
                    json.dumps(
                        self.model.profile_info.int8.to_dict(),
                        indent=4,
                    )
                )
                output.append("\n")
        if self.model and self.model.profile_info:
            output.append("Performance on device types:")
            output.append("============================")
            output.append(
                json.dumps(
                    self.model.profile_info.table.to_dict(),
                    indent=4,
                )
            )

        print("\n".join(output))

    @classmethod
    def from_dict(cls, obj: dict):
        """Create an instance of ProfileResponse from a dict."""
        return cls(**obj)


def profile(
    model: Union[Path, str, bytes, Any],
    device: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> ProfileResponse:
    """Profile the performance of a trained model on a range of embedded targets, or a specific device.

    The response includes estimates of memory usage and latency for the model across a range of
    targets, including low-end MCU, high-end MCU, high-end MCU with accelerator, microprocessor unit
    (MPU), and a GPU or neural network accelerator. It will also include details of any conditions
    that preclude operation on a given type of device.

    If you request a specific `device`, the results will also include estimates for that specific
    device. A list of devices can be obtained from `edgeimpulse.model.list_profile_devices()`.

    You can call `.summary()` on the response to obtain a more readable version of the most relevant
    information.

    Args:
        model (Union[Path, str, bytes, Any]): A machine learning model, or similarly represented
            computational graph. Can be `Path` or `str` denoting file path, Python `bytes`
            containing a model, or a Keras model instance.
        device (Optional[str], optional): An embedded processor for which to profile the model.
            A comprehensive list can be obtained via `edgeimpulse.model.list_profile_devices()`.
        api_key (Optional[str], optional): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or the env var
            `EI_API_KEY`.
        timeout_sec (Optional[float], optional): Number of seconds to wait for profile job to
            complete on the server. `None` is considered "infinite timeout" and will wait forever.

    Returns:
        ProfileResponse: Structure containing profile information.
        A subclass of `edgeimpulse_api.models.get_pretrained_model_response`.
        You can call its `.summary()` method for a more readable version of the
        most relevant information.

    Raises:
        Exception: Unhandled exception from API
        InvalidAuthTypeException: Incorrect authentication type was provided.
        InvalidDeviceException: Device is not valid.
        TimeoutException: Timeout waiting for result

    Examples:
        Profile a model on device

        ```python
        import edgeimpulse as ei #noqa: F401
        # ei.API_KEY = "<YOUR-KEY>" # or from env EI_API_KEY


        from edgeimpulse import model

        # Profile a Keras model across a range of devices
        result = model.profile(model=keras_model) # noqa: F821
        result.summary()

        # Profile different types of models on specific devices

        # ONNX
        result = model.profile(
            model="heart_rate.onnx",
            device="cortex-m4f-80mhz"
        )

        # TensorFlow SavedModel (can also be a zip)
        result = model.profile(
            model="heart_rate",
            device="nordic-nrf9160-dk"
        )

        # TensorFlow Lite (float32 or int8)
        result = model.profile(
            model="heart_rate.lite",
            device="synaptics-ka10000"
        )
        ```

    """
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    jobs = JobsApi(client)
    learn = LearnApi(client)

    project_id = default_project_id_for(client)

    if device:
        profile_devices = get_profile_devices(client, project_id)
        if device not in profile_devices:
            raise InvalidDeviceException(device, profile_devices)

    # The API bindings currently require files to be on disk.
    # We will write files to this temporary dir if necessary.
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            upload_pretrained_model_and_data(
                tempdir=tempdir,
                client=client,
                project_id=project_id,
                model=model,
                device=device,
                representative_data=None,
                timeout_sec=timeout_sec,
            )
        except TimeoutException as te:
            raise te
        except Exception as e:
            logging.debug(f"Exception uploading model [{str(e)}]")
            raise e

    # Start profiling job
    try:
        profile_response = learn.profile_pretrained_model(project_id)
        check_response_errors(profile_response)
        job_id = profile_response.id
    except Exception as e:
        logging.debug(f"Exception starting profile job [{str(e)}]")
        raise e

    # Wait for profile job to complete
    try:
        _ = poll(
            jobs_client=jobs,
            project_id=project_id,
            job_id=job_id,
            timeout_sec=timeout_sec,
        )
    except TimeoutException as te:
        raise te
    except Exception as e:
        raise e

    try:
        get_pretrained_model_response = learn.get_pretrained_model_info(project_id)
        check_response_errors(get_pretrained_model_response)
    except Exception as e:
        logging.debug(f"Exception retrieving profiling results [{str(e)}]")
        raise e

    profile_response = ProfileResponse.from_dict(
        get_pretrained_model_response.to_dict()
    )
    logging.debug(f"profile_response = {profile_response}")
    return profile_response


def list_profile_devices(api_key: Optional[str] = None) -> List[str]:
    """List possible values for the `device` field when calling `edgeimpulse.model.profile()`.

    Args:
        api_key (str, optional): The API key for an Edge Impulse project. This can also be set via
        the module-level variable `edgeimpulse.API_KEY`, or the env var `EI_API_KEY`.

    Returns:
        List[str]: List of profile targets for project

    """
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    return get_profile_devices(client)
