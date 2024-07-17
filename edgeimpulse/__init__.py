# Import the sub packages here to expose them to the user
# ruff: noqa: F401,D104
# mypy: ignore-errors
import os
import edgeimpulse.model
import edgeimpulse.datasets
import edgeimpulse.exceptions
import edgeimpulse.experimental

from edgeimpulse.util import configure_generic_client, default_project_id_for

import edgeimpulse_api

__version__ = "1.0.13"

try:
    API_KEY = os.environ["EI_API_KEY"]
except KeyError:
    API_KEY = None

try:
    EI_USERNAME = os.environ["EI_USERNAME"]
except KeyError:
    EI_USERNAME = None

try:
    EI_PASSWORD = os.environ["EI_PASSWORD"]
except KeyError:
    EI_PASSWORD = None

try:
    API_ENDPOINT = os.environ["EI_API_ENDPOINT"]
except KeyError:
    API_ENDPOINT = "https://studio.edgeimpulse.com/v1"

try:
    INGESTION_ENDPOINT = os.environ["EI_INGESTION_ENDPOINT"]
except KeyError:
    INGESTION_ENDPOINT = "https://ingestion.edgeimpulse.com"
