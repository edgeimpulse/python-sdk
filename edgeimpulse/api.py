"""Convenience API to initialize and access all Edge impulse."""

# Import the sub packages here to expose them to the user
# ruff: noqa: F401, D107
import os
import edgeimpulse.model
import edgeimpulse.exceptions
import edgeimpulse.experimental
from edgeimpulse.util import configure_generic_client, default_project_id_for

import edgeimpulse_api as edge_api


class EdgeImpulseApi:
    """Initialize the Edge Impulse Api.

    Args:
        host (str, optional): The host address. None will use the production host. Defaults to None
        key (str, optional): The authentication key to use. If none given, it will use no authentication.
        key_type (str, optional): The type of key. Can be `api`, `jwt` or `jwt_http`. Defaults to `api`.
    """

    user: edge_api.UserApi
    """Manage user activating, creation, updating and information"""

    classify: edge_api.ClassifyApi
    """Classify samples"""
    deployment: edge_api.DeploymentApi
    """Work with deployment targets"""
    devices: edge_api.DevicesApi
    """Work with devices in your project"""
    dsp: edge_api.DSPApi
    """Work with digital signal processing (feature extraction)"""
    export: edge_api.ExportApi
    """Export datasets and projects"""
    feature_flags: edge_api.FeatureFlagsApi
    """Enable and disable feature flags"""
    impulse: edge_api.ImpulseApi
    """Work and manage your impulse"""
    jobs: edge_api.JobsApi
    """Start and manage long running jobs"""
    learn: edge_api.LearnApi
    """Work with keras and pretrained models"""
    login: edge_api.LoginApi
    """Login and authenticate"""
    optimization: edge_api.OptimizationApi
    """Optimize the model with the eon tuner"""
    organization_blocks: edge_api.OrganizationBlocksApi
    """Work with organization blocks"""
    organization_create_project: edge_api.OrganizationCreateProjectApi
    """Automate project creation for organizations"""
    organization_data: edge_api.OrganizationDataApi
    """Work with organization data"""
    organization_data_campaigns: edge_api.OrganizationDataCampaignsApi
    """Work with organization campaigns"""
    organization_jobs: edge_api.OrganizationJobsApi
    """Start run and manage organization jobs"""
    organization_pipelines: edge_api.OrganizationPipelinesApi
    """Work with organization pipelines"""
    organization_portals: edge_api.OrganizationPortalsApi
    """Create and manage organization portals"""
    organizations: edge_api.OrganizationsApi
    """Work with your organizations"""
    performance_calibration: edge_api.PerformanceCalibrationApi
    """Calibrate your model with real world data"""
    projects: edge_api.ProjectsApi
    """Create and manage your projects"""
    raw_data: edge_api.RawDataApi
    """Work with your project data"""
    upload_portal: edge_api.UploadPortalApi
    """Create and manage data upload portals"""

    host: str
    "Edge Impulse studio host (defaults to production)"

    client: edge_api.ApiClient
    "The client used for initializing the apis, use `set_client` to update the client"

    def __init__(self, host: str = None, key: str = None, key_type: str = "api"):
        self.host = host
        config = edge_api.Configuration(self.host)
        if key is None:
            client = edge_api.ApiClient(config)
            self.set_client(client)
        else:
            self.authenticate(key=key, key_type=key_type)

    def default_project_id(self) -> int:
        """Retrieve the default project ID from the api key.

        Returns:
            int: The project associated with the api key.
        """
        return default_project_id_for(self.client)

    def authenticate(self, key: str, key_type: str = "api", host: str = None) -> None:
        """Authenticate against Edge Impulse.

        Args:
            key (str): The authentication key to use. If none give, it will use no authentication.
            key_type (str, optional): The type of key. Can be `api`, `jwt` or `jwt_http`. Defaults to `api`.
            host (str, optional): The host address. None will use the production host. Defaults to None
        """
        client = configure_generic_client(
            key=key, key_type=key_type, host=host or self.host
        )
        self.set_client(client)

    def set_client(self, client: edge_api.ApiClient) -> None:
        """Set the API client and initialize the APIs wit that client.

        Args:
            client: The API client.
        """
        self.client = client
        self.user = edge_api.UserApi(client)
        self.classify = edge_api.ClassifyApi(client)
        self.deployment = edge_api.DeploymentApi(client)
        self.devices = edge_api.DevicesApi(client)
        self.dsp = edge_api.DSPApi(client)
        self.export = edge_api.ExportApi(client)
        self.feature_flags = edge_api.FeatureFlagsApi(client)
        self.impulse = edge_api.ImpulseApi(client)
        self.jobs = edge_api.JobsApi(client)
        self.learn = edge_api.LearnApi(client)
        self.login = edge_api.LoginApi(client)
        self.optimization = edge_api.OptimizationApi(client)
        self.organization_blocks = edge_api.OrganizationBlocksApi(client)
        self.organization_create_project = edge_api.OrganizationCreateProjectApi(client)
        self.organization_data = edge_api.OrganizationDataApi(client)
        self.organization_data_campaigns = edge_api.OrganizationDataCampaignsApi(client)
        self.organization_jobs = edge_api.OrganizationJobsApi(client)
        self.organization_pipelines = edge_api.OrganizationPipelinesApi(client)
        self.organization_portals = edge_api.OrganizationPortalsApi(client)
        self.organizations = edge_api.OrganizationsApi(client)
        self.performance_calibration = edge_api.PerformanceCalibrationApi(client)
        self.projects = edge_api.ProjectsApi(client)
        self.raw_data = edge_api.RawDataApi(client)
        self.upload_portal = edge_api.UploadPortalApi(client)
