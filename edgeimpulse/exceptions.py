from typing import List, Optional


class EdgeImpulseException(Exception):
    def __init__(self, message):
        super().__init__(
            message
            + "\n\nFor more information see https://docs.edgeimpulse.com/reference/python-sdk"
            + " or ask a question at https://forum.edgeimpulse.com/"
        )


class InvalidDeviceException(EdgeImpulseException):
    """Exception raised when an invalid device is passed.

    Atrributes:
        device (str): device type to profile
        profile devices (List[str]): List of devices for a project as strings.

    """

    def __init__(self, device: str, profile_devices: List[str]):
        self.device = device
        self.profile_devices = profile_devices
        self.message = f"Invalid device: [{device}] valid types are: {profile_devices}"
        super().__init__(self.message)


class InvalidTargetException(EdgeImpulseException):
    """
        Exception raised when an invalid target is passed.
        For a list of valid targets use `edgeimpulse.model.list_deployment_targets()`.

    Atrributes:
        deploy_target (str): Target to deploy to.
        target_names (List[str]): List of targets for a project as strings.
    """

    def __init__(self, deploy_target: str, target_names: List[str]):
        self.deploy_target = deploy_target
        self.target_names = target_names
        self.message = f"deploy_target: [{deploy_target}] not in {target_names}"
        super().__init__(self.message)


class InvalidEngineException(EdgeImpulseException):
    """
    Exception raised when an invalid engine is passed.
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
                "API key was None, ensure you have set module level "
                "variable `edgeimpulse.API_KEY` or the environment "
                "variable `EI_API_KEY`. For help finding your "
                "API keys see https://docs.edgeimpulse.com/reference/edge-impulse-api/edge-impulse-api#api-key."
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
                "The Edge Impulse API responded with an error.\n"
                f"Error message: '{error}'"
                if error is not None
                else "There was no error message included."
            )
        )


class TimeoutException(EdgeImpulseException):
    """
    Exception raised when a timeout has been reached.
    """

    def __init__(self, msg: str):
        super().__init__(msg)
