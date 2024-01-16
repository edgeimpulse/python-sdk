import os

# Import the subpackages here to expose them to the user
# ruff: noqa: F401
import edgeimpulse.model
import edgeimpulse.exceptions
import edgeimpulse.experimental

__version__ = "1.0.8"

try:
    API_KEY = os.environ["EI_API_KEY"]
except KeyError:
    API_KEY = None

try:
    API_ENDPOINT = os.environ["EI_API_ENDPOINT"]
except KeyError:
    API_ENDPOINT = "https://studio.edgeimpulse.com/v1"

try:
    INGESTION_ENDPOINT = os.environ["EI_INGESTION_ENDPOINT"]
except KeyError:
    INGESTION_ENDPOINT = "https://ingestion.edgeimpulse.com"
