import logging
import tempfile
import time
import base64
from typing import Union, Optional, List, Tuple, Any
from pathlib import Path
import shutil
import os
from urllib3.exceptions import ReadTimeoutError
import zipfile

import edgeimpulse as ei

from edgeimpulse.exceptions import (
    InvalidAuthTypeException,
    MissingApiKeyException,
    InvalidModelException,
    UnsuccessfulRequestException,
    TimeoutException,
)

from edgeimpulse_api import (
    ApiClient,
    JobsApi,
    DeploymentApi,
    Configuration,
    ProjectsApi,
    GetJobResponse,
    LearnApi,
    RawDataApi,
)

DATA_CATEGORIES = ["training", "testing", "anomaly"]


def configure_generic_client(
    key: str,
    key_type: str = "api",
    host: str = "https://studio.edgeimpulse.com/v1",
) -> ApiClient:
    """Helper funtion to configure generic api client

    Args:
        key (str): api, jwt or jwt_http key.
        key_type (str, optional): Type of key. Defaults to 'api'.
        host (str): API host url. Defaults to "https://studio.edgeimpulse.com/v1".

    Raises:
        Exception: Unrecognized key_type

    Returns:
        ApiClient: Generic API client used in other generated APIs
    """

    if key is None:
        raise MissingApiKeyException()

    logging.debug(f"Using API host [{host}]")

    config = Configuration(host=host)
    if key_type == "api":
        config.api_key["ApiKeyAuthentication"] = key
    elif key_type == "jwt":
        config.api_key["JWTAuthentication"] = key
    elif key_type == "jwt_http":
        config.api_key["JWTHttpHeaderAuthentication"] = key
    else:
        msg = (
            f"Unrecognized key_type: {key_type},\n"
            'Valid key_types: "api", "jwt", "jwt_http"'
        )
        logging.error(msg)
        raise InvalidAuthTypeException(msg)

    client = ApiClient(config)
    # So we know which calls come from the SDK vs the client
    client.user_agent += f" edgeimpulse-sdk/{ei.__version__}"

    return client


def poll(
    jobs_client: JobsApi,
    project_id: int,
    job_id: int,
    timeout_sec: Optional[float] = None,
) -> GetJobResponse:
    """Helper function to syncronously poll a specific job within a project

    Args:
        jobs_client (JobsApi): JobsApi client
        project_id (int): Project id number
        job_id (int): Job id to poll
        timeout_sec (float, optional): Optional timeout for polling.

    Raises:
        e: Unhandled exception from api
        TimeoutException: Timeout waiting for result

    Returns:
        GetJobResponse: Structure containing job information

    """
    timeout = (time.time() + timeout_sec) if timeout_sec else None

    while True:
        try:
            job_response = jobs_client.get_job_status(project_id, job_id)
            check_response_errors(job_response)
            logging.info(f"Waiting for job {job_id} to finish...")
            if job_response.job.finished:
                logging.info(f"job_response = {job_response}")
                logs = jobs_client.get_jobs_logs(project_id, job_id)
                # TODO: parse logs so each stdout entry appears on new line
                logging.debug(
                    f"Logs for project: {project_id} " f"Job: {job_id}\n {logs}"
                )
                return job_response
            time.sleep(1)

        except Exception as e:
            # What are the set of exceptions we are willing to handle,
            # e.g. polling for job not finished
            if str(e) == "Job is still running":
                # Wait and continue
                time.sleep(1)
            elif str(e) == "Job did not finish successfully":
                logging.error(e)
                break
            elif str(e) == "Profile response is not available":
                logging.error(e)
                break
            # Everything else is propagated up
            else:
                logging.debug(f"Unhandled exception [{str(e)}]")
                raise e

        # See if polling timed out
        if timeout is not None and time.time() > timeout:
            # Log message
            err_msg = f"Timeout waiting for result for job_id {job_id}"
            logging.info(err_msg)

            # Cancel job
            try:
                logging.info(f"Canceling job {job_id}")
                cancel_response = jobs_client.cancel_job(
                    project_id, job_id, force_cancel="true"
                )
                check_response_errors(cancel_response)

            # Catch errors if the cancel request fails
            except Exception as e:
                # If unknown job ID, then the job is likely no longer running
                if "Unknown job ID " in str(e):
                    logging.error(e)
                    break

                # Everything else is propagated up
                else:
                    logging.debug(f"Unhandled exception [{str(e)}]")
                    raise e

            raise TimeoutException(err_msg)


# Try importing numpy (even if it isn't used, which triggers F401 in linting)
# ruff: noqa: F401
def numpy_installed():
    try:
        import numpy as np

        return True
    except ModuleNotFoundError:
        return False


# Try importing tensorflow (even if it isn't used, which triggers F401 in linting)
# ruff: noqa: F401
def tensorflow_installed():
    try:
        import tensorflow as tf

        return True
    except ModuleNotFoundError:
        return False


# Try importing onnx (even if it isn't used, which triggers F401 in linting)
# ruff: noqa: F401
def onnx_installed():
    try:
        import onnx

        return True
    except ModuleNotFoundError:
        return


def is_path_to_numpy_file(data):
    return is_type_accepted_by_open(data) and str(data).endswith(".npy")


def is_path_to_onnx_model(model):
    return is_type_accepted_by_open(model) and str(model).endswith(".onnx")


def is_type_accepted_by_open(model):
    return (isinstance(model, str)) or (issubclass(type(model), Path))


def is_path_to_tf_saved_model_zipped(model):
    if not is_type_accepted_by_open(model):
        return False
    if not os.path.exists(model):
        return False
    try:
        zip_file = zipfile.ZipFile(model)
        return "saved_model/saved_model.pb" in zip_file.namelist()
    except (zipfile.BadZipFile, IsADirectoryError):
        return False


def is_path_to_tf_saved_model_directory(model_dir):
    return os.path.exists(f"{model_dir}/saved_model.pb")


def encode_file_as_base64(filename: str):
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_keras_model(model):
    if not tensorflow_installed():
        return False
    import tensorflow as tf

    return issubclass(type(model), tf.keras.Model)


def is_onnx_model(model):
    if not onnx_installed():
        return False
    import onnx

    return issubclass(type(model), onnx.ModelProto)


def is_numpy_array(array):
    if not numpy_installed():
        return False
    import numpy as np

    return type(array) == np.ndarray


def make_zip_archive(saved_model_path):
    zip_path = shutil.make_archive(
        saved_model_path,
        "zip",
        root_dir=os.path.dirname(saved_model_path),
        base_dir="saved_model",
    )
    return zip_path


def save_model(model: Union[Path, str, bytes], directory: str) -> str:
    if isinstance(model, str) or isinstance(model, Path):
        raise Exception(f"Model is already located at path {model}")

    if tensorflow_installed() and is_keras_model(model):
        # Note: this needs to exactly match the format studio
        # expects for unpacking
        saved_model_path = os.path.join(directory, "saved_model")
        model.save(saved_model_path, save_format="tf")
        zip_path = make_zip_archive(saved_model_path)
        return zip_path
    if onnx_installed() and is_onnx_model(model):
        import onnx

        onnx_path = os.path.join(directory, "model.onnx")
        onnx.save(model, onnx_path)
        return onnx_path
    elif isinstance(model, bytes):
        filepath = os.path.join(directory, "model")
        with open(filepath, "wb") as f:
            f.write(model)
        return filepath

    raise InvalidModelException(
        f"Model was unexpected type {type(model)} and could not be processed"
    )


def inspect_model(model: Union[Path, str, bytes, Any], tempdir: str) -> Tuple[str, str]:
    """
    Helper function to load tflite model

    Args:
        model: Supports a number of ways of representing a model including
               1) A Path / str filename of an onnx file, a Keras
               saved_model.zip, a TensorFlow saved_model directory or saved
               tflite model. 2) a Keras model instance. 3) an ONNX model
               instance
        tempdir: temp dir used to write saved form of any in memory
                 model passed
    Returns:
        Tuple(str): (model type suitable for API,
                     path to model saved suitable for API)
    """
    try:
        if is_path_to_onnx_model(model):
            logging.info(f"Model parsed as ONNX (by path) [{model}]")
            return "onnx", model

        elif is_path_to_tf_saved_model_zipped(model):
            logging.info(f"Model parsed as SavedModel (by zip filename) [{model}]")
            return "saved_model", model

        elif is_path_to_tf_saved_model_directory(model):
            logging.info(f"Model parsed as SavedModel (by directory) [{model}]")

            root_dir, basename = os.path.split(model)
            if root_dir == "":
                # Make_archive wants an explicit .
                root_dir = "."

            if basename != "saved_model":
                # Upload wants a zip that has 'saved_model/' as the root
                # so, if this is not the case, work off a tmp copy so it will
                shutil.copytree(model, f"{tempdir}/saved_model")
                root_dir = tempdir
                basename = "saved_model"

            zip_base_filename = f"{tempdir}/saved_model"

            logging.info(
                f"Zipping; zip_base_filename=[{zip_base_filename}]"
                f" root_dir=[{root_dir}] base_dir=[saved_model]"
            )
            shutil.make_archive(
                zip_base_filename,
                "zip",
                root_dir=root_dir,
                base_dir="saved_model",
            )
            return "saved_model", f"{zip_base_filename}.zip"

        elif tensorflow_installed() and is_keras_model(model):
            logging.info("Model parsed as keras model (in memory)")
            saved_model_path = save_model(model, tempdir)
            return "saved_model", saved_model_path

        elif onnx_installed() and is_onnx_model(model):
            logging.info("Model parsed as ONNX model (in memory)")
            onnx_path = save_model(model, tempdir)
            return "onnx", onnx_path

        elif is_type_accepted_by_open(model):
            logging.info(f"Model parsed as assumed tflite file (on disk) [{model}]")
            return "tflite", model

        elif isinstance(model, bytes):
            logging.info("Model parsed as assumed tflite file (in memory)")
            saved_model_path = save_model(model, tempdir)
            return "tflite", saved_model_path

        else:
            raise Exception("Unexpected model type")

    except Exception as e:
        raise InvalidModelException(
            f"Was unable to load_model of type {type(model)}"
            f" with exception {str(e)}"
        ) from e


def inspect_representative_data(data: Union[Path, str, bytes, Any]) -> Optional[str]:
    """Ensure representative data is saved to disk for upload.

    Args:
        data: either a str/Path to a numpy array or a np.ndarray
        directory: a str with a directory path that the data should be saved to if necessary

    Returns:
        string path to the saved file

    """
    if data is None:
        return None

    if is_path_to_numpy_file(data):
        return str(data)

    if isinstance(data, str):
        raise Exception(
            f"Unknown representative data file {data}. Expecting file ending in .npy."
        )

    if numpy_installed():
        import numpy as np

        if type(data) == np.ndarray:
            return None

    raise Exception(
        f"Can't parse representative data. Expecting file ending in .npy but received {type(data)}."
    )


def save_representative_data(data: Union[Path, str, bytes], directory: str) -> str:
    if numpy_installed():
        import numpy as np

        if type(data) == np.ndarray:
            if directory is None:
                raise Exception(
                    "Directory must be specified if a numpy array is provided"
                )
            tmp_filename = f"{directory}/data.npy"
            np.save(tmp_filename, data)
            return tmp_filename

    raise Exception(f"Can't parse representative data. Unknown type {type(data)}")


def default_project_id_for(client: ApiClient) -> int:
    """Derive project id from api_key used to configure generic client

    Args:
        client (ApiClient): Generic api client configured with a project api key. For jwt or
            jwt_http key use get_project_id_list().

    Returns:
        int: Project id
    """
    projects = ProjectsApi(client)
    if any("JWT" in s for s in client.configuration.api_key.keys()):
        msg = (
            "This helper function is for use with project api_key only\n"
            "Use get_project_id_list() for api clients configured jwt or jwt_http key"
        )
        raise InvalidAuthTypeException(msg)

    try:
        response = projects.list_projects()
        check_response_errors(response)
        project_id = response.projects[0].id
    except Exception as e:
        logging.debug(f"Exception trying to fetch project ID [{str(e)}]")
        raise e

    logging.info(f"Derived project_id={project_id} based on api key")
    return project_id


def get_project_deploy_targets(
    client: ApiClient, project_id: Optional[int] = None
) -> List[str]:
    """Pull a list of deploy targets

    Args:
        client (ApiClient): Generic api client configured with a project api key

    Returns:
        List[str]: List of deploy targets for project
    """
    if project_id is None:
        project_id = default_project_id_for(client)
    try:
        deploy = DeploymentApi(client)
        response = deploy.list_deployment_targets_for_project_data_sources(project_id)
        check_response_errors(response)
        targets = response.targets
    except Exception as e:
        logging.debug(f"Exception trying to fetch project ID [{str(e)}]")
        raise e
    return [x.to_dict()["format"] for x in targets]


def get_profile_devices(
    client: ApiClient, project_id: Optional[int] = None
) -> List[str]:
    """Pull a list of profile devices

    Args:
        client (ApiClient): Generic api client configured with a project api key

    Returns:
        List[str]: List of profile targets for project
    """
    if project_id is None:
        project_id = default_project_id_for(client)
    try:
        projects = ProjectsApi(client)
        response = projects.get_project_info(project_id)
        check_response_errors(response)
        latency_devices = response.latency_devices
    except Exception as e:
        logging.debug(f"Exception trying to fetch project ID [{str(e)}]")
        raise e
    return [x.mcu for x in latency_devices]


def upload_pretrained_model_and_data(
    tempdir: str,
    client: ApiClient,
    project_id: int,
    model: Union[Path, str, bytes, Any],
    representative_data: Optional[Union[Path, str, bytes, Any]] = None,
    device: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> GetJobResponse:
    """
    Upload a model and data to Edge Impulse servers.

    Args:
        tempdir (str): Temporary directory to hold saved form of any model passed in
        client (ApiClient): Generic api client configured with a project api key. For jwt or
            jwt_http key use get_project_id_list().
        project_id (int): Project id number
        model (Union[Path, str, bytes, Any], op): A machine learning model, or similarly represented
            computational graph. Can be `Path` or `str` denoting file path, Python `bytes`
            containing a model, or a Keras model instance.
        representative_data (Optional[Union[Path, str, bytes, Any]], optional): A numpy
            representative input dataset. Accepts either an in memory numpy array or the Path/str
            filename of a np.save .npy file.
        device (Optional[str], optional): An embedded processor for which to profile the model.
            A comprehensive list can be obtained via `edgeimpulse.model.list_profile_devices()`.
        timeout_sec (Optional[float], optional): Optional timeout for polling.

    Raises:
        e: Unhandled exception from api
        FileNotFoundError: File or directory not found
        TimeoutException: Timeout waiting for result from server

    Returns:
        GetJobResponse: Structure containing job information
    """

    # Determine the type of model we have and make sure it is present on disk
    model_file_type, model_path = inspect_model(model, tempdir)

    # Determine the type of representative features we have and make sure they are
    # present on disk
    representative_features_path = inspect_representative_data(representative_data)
    if representative_data is not None and not representative_features_path:
        representative_features_path = save_representative_data(
            representative_data, tempdir
        )

    learn = LearnApi(client)
    jobs = JobsApi(client)

    try:
        response = learn.upload_pretrained_model(
            project_id=project_id,
            model_file=model_path,
            model_file_name=os.path.basename(model_path),
            model_file_type=model_file_type,
            representative_features=representative_features_path,
            device=device,
        )
        check_response_errors(response)
        job_id = response.id
    except FileNotFoundError as e:
        raise InvalidModelException(str(e)) from e
    except Exception as e:
        logging.debug(f"Exception starting upload job [{str(e)}]")
        raise e

    # Wait for upload job to complete
    try:
        job_response = poll(
            jobs_client=jobs,
            project_id=project_id,
            job_id=job_id,
            timeout_sec=timeout_sec,
        )
    except TimeoutException as te:
        raise te
    except Exception as e:
        raise e

    # Write out response
    logging.info(job_response)

    return job_response


def check_response_errors(request):
    """Checks for standard errors and raises an exception with the details if found."""
    if hasattr(request, "success"):
        success = request.success
        error = None
        if hasattr(request, "error"):
            error = request.error
        if success is not True:
            raise UnsuccessfulRequestException(error)
