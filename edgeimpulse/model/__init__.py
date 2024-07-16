"""Use this module to profile, and deploy your edge models."""

__all__ = [
    "profile",
    "list_profile_devices",
    "deploy",
    "list_engines",
    "list_model_types",
    "list_deployment_targets",
]
from edgeimpulse.model._functions.profile import profile, list_profile_devices
from edgeimpulse.model._functions.deploy import (
    deploy,
    list_engines,
    list_model_types,
    list_deployment_targets,
)
