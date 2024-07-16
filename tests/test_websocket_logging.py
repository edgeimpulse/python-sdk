# ruff: noqa: D102, D101, D100
import unittest
import logging
import os

from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
    run_organization_job_until_completion,
    run_project_job_until_completion,
)

import edgeimpulse as ei

from tests.util import delete_all_samples

from edgeimpulse_api import (
    OrganizationDataApi,
    JobsApi,
    ExportOriginalDataRequest,
)

logging.getLogger().setLevel(logging.INFO)


class TestWebsocketLogging(unittest.TestCase):
    def test_not_exist(self):
        with self.assertRaises(Exception) as context:
            run_project_job_until_completion(999999)
        self.assertIn("No job with this ID found", str(context.exception))

    def test_project_job(self):
        delete_all_samples()

        res = ei.experimental.data.upload_directory(
            directory="tests/sample_data/gestures", category="testing"
        )

        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)

        client = configure_generic_client(key=ei.API_KEY, host=ei.API_ENDPOINT)

        project_id = default_project_id_for(client)
        jobs = JobsApi(client)

        res = jobs.start_original_export_job(
            project_id=project_id,
            export_original_data_request=ExportOriginalDataRequest(
                retainCrops=True, uploaderFriendlyFilenames=False
            ),
        )
        self.assertEqual(res.success, True)
        run_project_job_until_completion(res.id)

        ## CHECK TIMEOUTS

        res = jobs.start_original_export_job(
            project_id=project_id,
            export_original_data_request=ExportOriginalDataRequest(
                retainCrops=True, uploaderFriendlyFilenames=False
            ),
        )

        with self.assertRaises(Exception) as context:
            run_project_job_until_completion(res.id, timeout_sec=3)

        self.assertIn(
            f"Timeout reached while waiting for job {res.id}", str(context.exception)
        )

    def test_org_job(self):
        if os.environ.get("RUN_WEBSOCKET_ORG", None) is None:
            return

        org_id = int(os.environ.get("EI_ORGANIZATION_ID"))
        org_api_key = os.environ.get("EI_ORGANIZATION_API_KEY")

        client = configure_generic_client(key=org_api_key, host=ei.API_ENDPOINT)

        data_api = OrganizationDataApi(client)
        res = data_api.organization_bulk_update_metadata(
            organization_id=org_id,
            dataset="my-dataset",
            csv_file="tests/sample_data/metadata.csv",
        )

        run_organization_job_until_completion(
            organization_id=org_id, job_id=res.id, client=client
        )

        print(res)
