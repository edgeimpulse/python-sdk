"""Convenience API to initialize and access all Edge impulse."""

# Import the sub packages here to expose them to the user
# ruff: noqa: F401, D107
import os
import edgeimpulse.model
import edgeimpulse.exceptions
import edgeimpulse.experimental
from edgeimpulse.util import configure_generic_client, default_project_id_for
import edgeimpulse as ei
import edgeimpulse_api as edge_api
from typing import Optional
from edgeimpulse.util import (
    run_project_job_until_completion as run_project_job_until_completion_util,
    run_organization_job_until_completion as run_organization_job_until_completion_util,
)

# The whole setup here with properties and internal variables using the __ dunder methods is in order to
# get autocomplete in vscode and spinx docs


class EdgeImpulseApi:
    """Initialize the Edge Impulse Api.

    Args:
        host (str, optional): The host address. None will use the production host. Defaults to None
        key (str, optional): The authentication key to use. If none given, it will use no authentication.
        key_type (str, optional): The type of key. Can be `api`, `jwt` or `jwt_http`. Defaults to `api`.
    """

    @property
    def user(self) -> edge_api.UserApi:
        """Manage user activation, creation and updates."""
        return self.__user

    @property
    def classify(self) -> edge_api.ClassifyApi:
        """Classify samples in your project."""
        return self.__classify

    @property
    def deployment(self) -> edge_api.DeploymentApi:
        """Work with your model deployment targets."""
        return self.__deployment

    @property
    def devices(self) -> edge_api.DevicesApi:
        """Work with devices in your project."""
        return self.__devices

    @property
    def dsp(self) -> edge_api.DSPApi:
        """Work with (DSP) digital signal processing and feature extraction blocks in your project."""
        return self.__dsp

    @property
    def export(self) -> edge_api.ExportApi:
        """Export your project."""
        return self.__export

    @property
    def impulse(self) -> edge_api.ImpulseApi:
        """Work and manage your Impulse (on-device feature extraction and classification pipeline)."""
        return self.__impulse

    @property
    def jobs(self) -> edge_api.JobsApi:
        """Start and manage long running jobs."""
        return self.__jobs

    @property
    def learn(self) -> edge_api.LearnApi:
        """Work with keras and pretrained models."""
        return self.__learn

    @property
    def login(self) -> edge_api.LoginApi:
        """Login and authenticate."""
        return self.__login

    @property
    def optimization(self) -> edge_api.OptimizationApi:
        """Optimize and find a better model with the EON tuner."""
        return self.__optimization

    @property
    def organization_blocks(self) -> edge_api.OrganizationBlocksApi:
        """Work with organization blocks."""
        return self.__organization_blocks

    @property
    def organization_create_project(self) -> edge_api.OrganizationCreateProjectApi:
        """Automate project creation for organizations."""
        return self.__organization_create_project

    @property
    def organization_data(self) -> edge_api.OrganizationDataApi:
        """Work with organization data."""
        return self.__organization_data

    @property
    def organization_data_campaigns(self) -> edge_api.OrganizationDataCampaignsApi:
        """Work with organization data campaigns."""
        return self.__organization_data_campaigns

    @property
    def organization_jobs(self) -> edge_api.OrganizationJobsApi:
        """Start and manage organization jobs."""
        return self.__organization_jobs

    @property
    def organization_pipelines(self) -> edge_api.OrganizationPipelinesApi:
        """Work with organization pipelines."""
        return self.__organization_pipelines

    @property
    def organization_portals(self) -> edge_api.OrganizationPortalsApi:
        """Create and manage organization portals."""
        return self.__organization_portals

    @property
    def organizations(self) -> edge_api.OrganizationsApi:
        """Work with your organizations."""
        return self.__organizations

    @property
    def performance_calibration(self) -> edge_api.PerformanceCalibrationApi:
        """Calibrate your model performance with real world data."""
        return self.__performance_calibration

    @property
    def projects(self) -> edge_api.ProjectsApi:
        """Create and manage your projects."""
        return self.__projects

    @property
    def raw_data(self) -> edge_api.RawDataApi:
        """Work with your project data."""
        return self.__raw_data

    @property
    def upload_portal(self) -> edge_api.UploadPortalApi:
        """Create and manage data upload portals."""
        return self.__upload_portal

    @property
    def host(self) -> Optional[str]:
        """Edge Impulse studio host (defaults to production)."""
        return self.__host  # type: ignore

    @property
    def client(self) -> edge_api.ApiClient:
        """The client used for initializing the apis, use `set_client` to update the client."""
        return self.__client

    def __init__(
        self,
        host: Optional[str] = None,
        key: Optional[str] = None,
        key_type: str = "api",
    ):
        self.__host = host or ei.API_ENDPOINT
        key = key or ei.API_KEY
        config = edge_api.Configuration(self.__host)
        if key is None:
            client = edge_api.ApiClient(config)
            self.set_client(client)
        else:
            self.authenticate(key=key, key_type=key_type)

    def run_project_job_until_completion(
        self,
        job_id: int,
        data_cb=None,
        client=None,
        project_id: Optional[int] = None,
        timeout_sec: Optional[int] = None,
    ) -> None:
        """Runs a project job until completion.

        Args:
            job_id (int): The ID of the job to run.
            data_cb (callable, optional): Callback function to handle job data. Use `lambda line: print(line)` to print logs.
            client (object, optional): An API client object. If None, a generic client will be configured.
            project_id (int, optional): The ID of the project. If not provided, a default project ID will be used.
            timeout_sec (int, optional): Number of seconds before timing out the job with an exception. Default is None

        Returns:
            None
        """
        if client is None:
            client = self.__client

        return run_project_job_until_completion_util(
            job_id=job_id,
            client=client,
            data_cb=data_cb,
            project_id=project_id,
            timeout_sec=timeout_sec,
        )

    def run_organization_job_until_completion(
        self,
        organization_id: int,
        job_id: int,
        data_cb=None,
        client=None,
        timeout_sec: Optional[int] = None,
    ) -> None:
        """Runs an organization job until completion.

        Args:
            organization_id (int): The ID of the organization.
            job_id (int): The ID of the job to run.
            data_cb (callable, optional): Callback function to handle job data.
            client (object, optional): An API client object. If None, a generic client will be configured.
            timeout_sec (int, optional): Number of seconds before timing out the job with an exception. Default is None.

        Returns:
            None
        """
        if client is None:
            client = self.__client

        return run_organization_job_until_completion_util(
            job_id=job_id,
            client=client,
            data_cb=data_cb,
            organization_id=organization_id,
            timeout_sec=timeout_sec,
        )

    def default_project_id(self) -> int:
        """Get the default project ID from the provided API key.

        Returns:
            int: The project associated with the api key.
        """
        return default_project_id_for(self.__client)

    def authenticate(
        self, key: str, key_type: str = "api", host: Optional[str] = None
    ) -> None:
        """Authenticate against Edge Impulse.

        Args:
            key (str): The authentication key to use. If none give, it will use no authentication.
            key_type (str, optional): The type of key. Can be `api`, `jwt` or `jwt_http`. Defaults to `api`.
            host (str, optional): The host address. None will use the production host. Defaults to None
        """
        host_url = host or self.__host
        client = configure_generic_client(key=key, key_type=key_type, host=host_url)
        self.set_client(client)

    def set_client(self, client: edge_api.ApiClient) -> None:
        """Set the API client and initialize the APIs wit that client.

        Args:
            client: The API client.
        """
        self.__client = client
        self.__user = edge_api.UserApi(client)
        self.__classify = edge_api.ClassifyApi(client)
        self.__deployment = edge_api.DeploymentApi(client)
        self.__devices = edge_api.DevicesApi(client)
        self.__dsp = edge_api.DSPApi(client)
        self.__export = edge_api.ExportApi(client)
        self.__impulse = edge_api.ImpulseApi(client)
        self.__jobs = edge_api.JobsApi(client)
        self.__learn = edge_api.LearnApi(client)
        self.__login = edge_api.LoginApi(client)
        self.__optimization = edge_api.OptimizationApi(client)
        self.__organization_blocks = edge_api.OrganizationBlocksApi(client)
        self.__organization_create_project = edge_api.OrganizationCreateProjectApi(
            client
        )
        self.__organization_data = edge_api.OrganizationDataApi(client)
        self.__organization_data_campaigns = edge_api.OrganizationDataCampaignsApi(
            client
        )
        self.__organization_jobs = edge_api.OrganizationJobsApi(client)
        self.__organization_pipelines = edge_api.OrganizationPipelinesApi(client)
        self.__organization_portals = edge_api.OrganizationPortalsApi(client)
        self.__organizations = edge_api.OrganizationsApi(client)
        self.__performance_calibration = edge_api.PerformanceCalibrationApi(client)
        self.__projects = edge_api.ProjectsApi(client)
        self.__raw_data = edge_api.RawDataApi(client)
        self.__upload_portal = edge_api.UploadPortalApi(client)
