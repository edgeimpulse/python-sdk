# ruff: noqa: D100, D101, D107
from typing import List, Optional


class EdgeImpulseException(Exception):
    def __init__(self, message):
        super().__init__(
            "\r\n"
            + "----------------"
            + "\r\n"
            + message
            + "\r\n"
            + "----------------"
            + "\r\n\r\nFor more information see https://docs.edgeimpulse.com/reference/python-sdk"
            + " or ask a question at https://forum.edgeimpulse.com/"
        )


class InvalidDeviceException(EdgeImpulseException):
    """Exception raised when an invalid device is passed.

    Attributes:
        device (str): device type to profile
        profile devices (List[str]): List of devices for a project as strings.

    """

    def __init__(self, device: str, profile_devices: List[str]):
        self.device = device
        self.profile_devices = profile_devices
        self.message = f"Invalid device: [{device}] valid types are: {profile_devices}"
        super().__init__(self.message)


class InvalidTargetException(EdgeImpulseException):
    """Exception raised when an invalid target is passed.

    For a list of valid targets use `edgeimpulse.model.list_deployment_targets()`.

    Attributes:
        deploy_target (str): Target to deploy to.
        target_names (List[str]): List of targets for a project as strings.
    """

    def __init__(self, deploy_target: str, target_names: List[str]):
        self.deploy_target = deploy_target
        self.target_names = target_names
        self.message = f"deploy_target: [{deploy_target}] not in {target_names}"
        super().__init__(self.message)


class InvalidEngineException(EdgeImpulseException):
    """Exception raised when an invalid engine is passed.

    For a list of valid engines use `edgeimpulse.model.list_engines()`.

    """

    def __init__(self, validation_error):
        super().__init__(str(validation_error))


class InvalidDeployParameterException(EdgeImpulseException):
    """Exception raised when an invalid parameter is passed."""

    def __init__(self, msg: str):
        super().__init__(msg)


class InvalidAuthTypeException(EdgeImpulseException):
    pass


class MissingApiKeyException(EdgeImpulseException):
    def __init__(self):
        super().__init__(
            (
                "API key was `None`, ensure you have set module level "
                "variable `edgeimpulse.API_KEY` or the environment "
                "variable `EI_API_KEY`. \r\nFor help finding your "
                "API keys see https://docs.edgeimpulse.com/reference/edge-impulse-api/edge-impulse-api#api-key."
            )
        )


class MissingApiIngestionEndpointException(EdgeImpulseException):
    def __init__(self):
        super().__init__(
            (
                "INGESTION_ENDPOINT was `None`, ensure you have set module level "
                "variable `edgeimpulse.INGESTION_ENDPOINT` or the environment variable `EI_INGESTION_ENDPOINT`"
            )
        )


class InvalidModelException(EdgeImpulseException):
    def __init__(self, msg: str):
        super().__init__(msg)


class UnsuccessfulRequestException(EdgeImpulseException):
    def __init__(self, error: Optional[str]):
        self.error = error
        super().__init__(
            (
                "The Edge Impulse API responded with an error.\r\n"
                f"Error message: '{error}'"
                if error is not None
                else "There was no error message included."
            )
        )


class TimeoutException(EdgeImpulseException):
    """Exception raised when a timeout has been reached."""

    def __init__(self, msg: str):
        super().__init__(msg)


class UnsupportedSampleType(EdgeImpulseException):
    """Exception raised when attempting to upload or download a data type that is not supported by Edge Impulse."""

    def __init__(self, error: Optional[str]):
        self.error = error
        super().__init__(
            (
                "Unsupported sample type. See here to learn more about the "
                "supported sample types: "
                "https://docs.edgeimpulse.com/reference/data-acquisition-format"
                f"Error message: '{error}'"
                if error is not None
                else "There was no error message included."
            )
        )
