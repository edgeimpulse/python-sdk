import logging
import re
import os
import tempfile
import io
from pathlib import Path

from typing import Union, Optional, Any, List

import edgeimpulse
import pydantic

from edgeimpulse.model.output_type import (
    Classification,
    Regression,
    ObjectDetection,
)
from edgeimpulse.model.input_type import (
    ImageInput,
    AudioInput,
    TimeSeriesInput,
    OtherInput,
)
from edgeimpulse.exceptions import (
    InvalidTargetException,
    InvalidEngineException,
    InvalidDeployParameterException,
    EdgeImpulseException,
    InvalidModelException,
    TimeoutException,
)

from edgeimpulse.util import (
    configure_generic_client,
    poll,
    default_project_id_for,
    get_project_deploy_targets,
    upload_pretrained_model_and_data,
    check_response_errors,
)
from edgeimpulse_api import JobsApi, DeploymentApi, LearnApi
from edgeimpulse_api.models.save_pretrained_model_request import (
    SavePretrainedModelRequest,
)
from edgeimpulse_api.models.build_on_device_model_request import (
    BuildOnDeviceModelRequest,
)
from edgeimpulse_api.models.deployment_target_engine import DeploymentTargetEngine
from edgeimpulse_api.models.keras_model_type_enum import KerasModelTypeEnum
from edgeimpulse_api.models.pretrained_model_tensor import PretrainedModelTensor


def deploy(
    model: Union[Path, str, bytes, Any],
    model_output_type: Union[Classification, Regression, ObjectDetection],
    model_input_type: Optional[
        Union[ImageInput, AudioInput, TimeSeriesInput, OtherInput]
    ] = None,
    representative_data_for_quantization: Optional[Union[Path, str, bytes, Any]] = None,
    deploy_model_type: Optional[str] = None,
    engine: str = "tflite",
    deploy_target: str = "zip",
    output_directory: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> io.BytesIO:
    """Transforms a machine learning model into a library for an edge device

    Transforms a trained model into a library, package, or firmware ready to deploy on an embedded
    device. Can optionally apply post-training quantization if a representative data sample is
    uploaded.

    Supported model formats:

    * `Keras Model instance <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_
    * `TensorFlow SavedModel <https://www.tensorflow.org/guide/saved_model>`_ (as path to directory
        or `.zip` file)
    * `ONNX model file <https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-onnx-model>`_
        (as path to `.onnx` file)
    * `TensorFlow Lite file <https://www.tensorflow.org/lite/guide>`_ (as bytes, or path to any file
        that is not `.zip` or `.onnx`)

    Representative data for quantization:

    * Must be a numpy array or `.npy` file.
    * Each element must have the same shape as your model's input.
    * Must be representative of the range (maximum and minimum) of values in your training data.

    Note: the available deployment options will change depending on the values given
    for `model`, `model_output_type`, and `model_input_type`. For example, the `openmv`
    deployment option is only available if `model_input_type` is set to `ImageInput`. If
    you attempt to deploy to an unavailable target, you will receive the error `Could
    not deploy: deploy_target: ...`.

    Args:
        model (Union[Path, str, bytes, Any]): A machine learning model, or similarly represented
            computational graph. Can be `Path` or `str` denoting file path, Python `bytes`
            containing a model, or a Keras model instance.
        model_output_type (Union[Classification, Regression, ObjectDetection]): Describe your
            model's type:Classification, Regression, or ObjectDetection. The types are available in
            the module `edgeimpulse.model.output_type`.
        model_input_type (Union[ImageInput, AudioInput, TimeSeriesInput, OtherInput], optional):
            Determines any input preprocessing (windowing, downsampling) that should be performed by
            the resulting library. The types are available in `edgeimpulse.model.input_type`. The
            default is `OtherInput` (no preprocessing).
        representative_data_for_quantization: A numpy representative input dataset. Accepts either
            an in memory numpy array or the Path/str filename of a np.save .npy file.
        deploy_model_type (str, optional): Use `int8` to receive an 8-bit quantized model, `float32`
            for non-quantized. Defaults to None, in which case it will become `int8` if
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
        timeout_sec (Optional[float], optional): Number of seconds to wait for profile job to
            complete on the server. `None` is considered "infinite timeout" and will wait forever.

    Returns:
        BytesIO: A stream containing a binary representation of the deployment output.

    Raises:
        InvalidAuthTypeException: Incorrect authentication type was provided.
        InvalidDeployParameterException: Unacceptable parameter given to deploy function.
        InvalidEngineException: Unacceptable engine for this target.
        InvalidTargetException: Unacceptable deploy_target for this project.
        FileNotFoundError: Model file could not be loaded.
        TimeoutException: Timeout waiting for result
        Exception: Unhandled exception from API

    Examples:

        .. code-block:: python

            # Turn a Keras model into a C++ library and write to disk
            ei.model.deploy(model=keras_model,
                            model_output_type=ei.model.output_type.Classification(),
                            model_input_type=ei.model.input_type.OtherInput(),
                            output_directory=".")

            # Convert various types of serialized models:
            ei.model.deploy(model="heart_rate.onnx", # ONNX
                            model_output_type=ei.model.output_type.Regression())
            ei.model.deploy(model="heart_rate", # TensorFlow SavedModel (can also be a zip)
                            model_output_type=ei.model.output_type.Regression())
            ei.model.deploy(model="heart_rate.lite", # TensorFlow Lite
                            model_output_type=ei.model.output_type.Regression())

            # Quantize a model to int8 during deployment by passing a numpy array of data
            ei.model.deploy(model=keras_model,
                            representative_data_for_quantization=x_test,
                            model_output_type=ei.model.output_type.Classification(),
                            output_directory=".")

            # The function returns a BytesIO which can be written as desired
            output = ei.model.deploy(model=keras_model,
                                     model_output_type=ei.model.output_type.Classification())
            with open('destination.zip', 'wb') as f:
                f.write(output.read())

    """

    if model_input_type is None:
        model_input_type = OtherInput()

    if deploy_model_type is not None and deploy_model_type not in list_model_types():
        raise InvalidDeployParameterException(
            "deploy_model_type must be None, or one of the following:\n"
            f"{list_model_types()}\n"
            " If None and representative_data_for_quantization is specified,"
            " then int8 will be used, otherwise float32 is assumed.\n"
            "For a list of valid model types use `edgeimpulse.model.list_model_types()`."
        )

    if engine not in list_engines():
        raise InvalidEngineException(
            f"Engine '{engine}' is not valid. It must be one of the following:\n"
            f"{list_engines()}"
        )

    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)

    # The API bindings currently require files to be on disk.
    # We will write files to this temporary dir if necessary.
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            upload_pretrained_model_and_data(
                tempdir=tempdir,
                client=client,
                project_id=project_id,
                model=model,
                representative_data=representative_data_for_quantization,
                timeout_sec=timeout_sec,
            )
        except TimeoutException as te:
            raise te
        except Exception as e:
            logging.debug(f"Exception uploading model [{str(e)}]")
            raise e

    learn = LearnApi(client)
    jobs = JobsApi(client)
    deploy = DeploymentApi(client)

    # Start fetching model job
    try:
        response = learn.get_pretrained_model_info(project_id=project_id)
        check_response_errors(response)
        available_model_types = response.available_model_types
        if response.model is None:
            raise EdgeImpulseException(
                "get_pretrained_model_info did not return model details."
            )
        outputs = response.model.outputs
    except Exception as e:
        logging.debug(f"Exception fetching model info: [{str(e)}]")
        raise e

    deploy_model_type = _determine_deploy_type(
        deploy_model_type=deploy_model_type,
        representative_data_for_quantization=representative_data_for_quantization,
        available_model_types=available_model_types,
    )

    model_output_type = _determine_output_type(
        model_output_type=model_output_type, outputs=outputs
    )

    try:
        r = SavePretrainedModelRequest.from_dict(
            {"input": model_input_type, "model": model_output_type}
        )
        response = learn.save_pretrained_model_parameters(
            project_id=project_id, save_pretrained_model_request=r
        )
        check_response_errors(response)
    except Exception as e:
        logging.debug(f"Exception calling save_pretrained_model_parameters [{str(e)}]")
        raise e

    target_names = get_project_deploy_targets(client, project_id=project_id)
    if deploy_target not in target_names:
        raise InvalidTargetException(deploy_target, target_names)

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
        response = jobs.build_on_device_model_job(
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
            jobs_client=jobs,
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
        response = deploy.download_build(
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


def list_deployment_targets(api_key: Optional[str] = None) -> "List[str]":
    """
    Lists suitable deployment targets for the project associated with configured or provided api
    key.

    Args:
        api_key (str, optional): The API key for an Edge Impulse project.
            This can also be set via the module-level variable `edgeimpulse.API_KEY`, or the env var
            `EI_API_KEY`.

    Returns:
        List[str]: List of deploy targets for project

    """
    client = configure_generic_client(
        key=api_key if api_key else edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    return get_project_deploy_targets(client)


def list_engines() -> "List[str]":
    """Lists all the engines that can be passed to `deploy()`'s `engine` parameter.

    Returns:
        List[str]: List of engines

    """
    return [e.value for e in DeploymentTargetEngine]


def list_model_types() -> "List[str]":
    """Lists all the model types that can passed to `deploy()`'s `deploy_model_type` parameter.

    Returns:
        List[str]: List of model types

    """
    return [t.value for t in KerasModelTypeEnum]


def _determine_deploy_type(
    deploy_model_type: Optional[str],
    representative_data_for_quantization: Optional[Union[Path, str, bytes, Any]],
    available_model_types: List[KerasModelTypeEnum],
):
    if deploy_model_type is not None and deploy_model_type not in available_model_types:
        raise InvalidDeployParameterException(
            f"You specified a deploy_model_type of {deploy_model_type}, but "
            f"for this model only these are available:\n"
            f"{str(available_model_types)}"
        )

    # Depending on whether a representative dataset has been provided we assume float32 or
    # switch to int8, but never clobber user requested one ( if provided )
    if representative_data_for_quantization is None:
        if deploy_model_type is None:
            logging.info(
                "Both representative_data_for_quantization &"
                " deploy_model_type are None so setting"
                " deploy_model_type to float32"
            )
            # We may have an int8 model
            if "float32" in available_model_types:
                deploy_model_type = "float32"
            else:
                if "int8" in available_model_types:
                    deploy_model_type = "int8"
                else:
                    raise InvalidDeployParameterException(
                        f"You did not specify a deploy_model_type and we "
                        "were unable to determine it automatically. Acceptable"
                        "values for this model are:\n"
                        f"{str(available_model_types)}"
                    )
    else:
        if deploy_model_type is None:
            logging.info(
                "Both representative_data_for_quantization provided &"
                " deploy_model_type is None so setting"
                " deploy_model_type to int8"
            )
            if "int8" in available_model_types:
                deploy_model_type = "int8"
            else:
                raise InvalidDeployParameterException(
                    f"You provided representative_data_for_quantization, "
                    "which implies an int8 deploy_model_type, but for this "
                    "model int8 is not available. Available types are:\n"
                    f"{str(available_model_types)}"
                )

    return deploy_model_type


def _determine_output_type(
    model_output_type: Union[Classification, Regression, ObjectDetection],
    outputs: List[PretrainedModelTensor],
):
    # Validate the specified model output as much as possible
    if type(model_output_type) != ObjectDetection:
        if len(outputs) > 1:
            raise InvalidModelException(
                f"Expected {type(model_output_type)} model to have 1 "
                f"output but it has {len(outputs)}"
            )
        if len(outputs[0].shape) != 2:
            raise InvalidModelException(
                f"Expected {type(model_output_type)} model to have 2 output "
                f"dimensions but has {len(outputs[0].shape)}"
            )
        if type(model_output_type) == Regression:
            output_neurons = outputs[0].shape[1]
            if output_neurons > 1:
                raise InvalidModelException(
                    f"Expected Regression model to have scalar output but "
                    f"has vector with length {output_neurons}"
                )

    # Allow for setting the number of labels
    if type(model_output_type) == Classification:
        output_neurons = outputs[0].shape[1]
        if model_output_type["labels"] is None:
            logging.info(
                f"Setting labels to match model output length of {output_neurons}"
            )
            model_output_type["labels"] = [str(num) for num in range(output_neurons)]
        else:
            expected_neurons = len(model_output_type["labels"])
            if output_neurons != len(model_output_type["labels"]):
                raise InvalidDeployParameterException(
                    "You specified a Classification model with "
                    f"{expected_neurons} labels but the model has "
                    f"{output_neurons} labels."
                )

    return model_output_type
