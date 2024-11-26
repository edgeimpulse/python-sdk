# ruff: noqa: D100, D101, D102, D103
import logging
import os
import tempfile
import unittest
import zipfile

import edgeimpulse as ei

from edgeimpulse import data
from edgeimpulse.experimental.impulse import build

from edgeimpulse.exceptions import (
    TimeoutException,
)
from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
    poll,
)

from edgeimpulse_api import (
    GenerateFeaturesRequest,
    Impulse,
    ImpulseApi,
    JobsApi,
    ProjectsApi,
    SetKerasParameterRequest,
    UpdateProjectRequest,
)

from tests.util import (
    assert_uploaded_samples,
    delete_all_samples,
)

logging.getLogger().setLevel(logging.INFO)

# How long to wait (seconds) for jobs to complete
JOB_TIMEOUT = 3600.0  # 60 min


class TestImpulse(unittest.TestCase):
    """Test impulse functions."""

    def setUp(self):
        delete_all_samples()

    def tearDown(self):
        delete_all_samples()

    def verify_valid_zip_file(self, fname=None, data=None):
        # verify zip; will throw exception on ZipFile creation
        # and return None if zip has valid checksums etc
        if fname:
            if zipfile.ZipFile(fname).testzip() is not None:
                raise Exception("Bad zipfile created by deploy")

        if data:
            if zipfile.ZipFile(data).testzip() is not None:
                raise Exception("Bad zip data created by deploy")

        if fname and data:
            # Rewind the stream
            data.seek(0)
            with open(fname, "rb") as f:
                contents = f.read()
                self.assertEqual(
                    data.read(), contents, "Zip in memory and on disk are not the same"
                )

    def get_only_file(self, dirname):
        """Return the only file in a directory."""
        files = os.listdir(dirname)
        self.assertEqual(len(files), 1)
        return os.path.join(dirname, files[0])

    def test_build(self):
        # Upload a dataset
        res = data.upload_directory(
            directory="tests/sample_data/gestures",
            metadata={"device": "phone"},
        )
        self.assertEqual(len(res.successes), 26)
        self.assertEqual(len(res.fails), 0)
        assert_uploaded_samples(self, res.successes)

        # Get API clients
        client = configure_generic_client(
            key=ei.API_KEY,
            host=ei.API_ENDPOINT,
        )
        impulse_api = ImpulseApi(client)
        jobs_api = JobsApi(client)
        projects_api = ProjectsApi(client)

        # Get project ID
        project_id = default_project_id_for(client)

        # Create an impulse
        dsp_id = 2
        learn_id = 3

        impulse = Impulse.from_dict(
            {
                "inputBlocks": [
                    {
                        "id": 1,
                        "type": "time-series",
                        "name": "Time series",
                        "title": "Time series data",
                        "windowSizeMs": 1000,
                        "windowIncreaseMs": 500,
                        "frequencyHz": 62.5,
                        "padZeros": True,
                    }
                ],
                "dspBlocks": [
                    {
                        "id": dsp_id,
                        "type": "spectral-analysis",
                        "name": "Spectral Analysis",
                        "title": "processing",
                        "axes": ["accX", "accY", "accZ"],
                        "input": 1,
                    }
                ],
                "learnBlocks": [
                    {
                        "id": learn_id,
                        "type": "keras",
                        "name": "Classifier",
                        "title": "Classification",
                        "dsp": [dsp_id],
                    }
                ],
            }
        )

        # Make sure the project is not in BYOM mode
        update_project_request = UpdateProjectRequest.from_dict(
            {"inPretrainedModelFlow": False}
        )

        response = projects_api.update_project(
            project_id=project_id,
            update_project_request=update_project_request,
        )
        if not hasattr(response, "success") or getattr(response, "success") is False:
            raise RuntimeError("Could not update project to impulse mode.")

        # Delete current impulse
        response = impulse_api.delete_impulse(project_id=project_id)
        if not hasattr(response, "success") or getattr(response, "success") is False:
            raise RuntimeError("Could not delete current impulse.")

        # Add blocks to impulse
        response = impulse_api.create_impulse(project_id=project_id, impulse=impulse)
        if not hasattr(response, "success") or getattr(response, "success") is False:
            raise RuntimeError("Could not create dummy impulse.")

        # Define generate features request
        generate_features_request = GenerateFeaturesRequest.from_dict(
            {
                "dspId": dsp_id,
                "calculate_feature_importance": False,
                "skip_feature_explorer": True,
            }
        )

        # Generate features
        response = jobs_api.generate_features_job(
            project_id=project_id,
            generate_features_request=generate_features_request,
        )
        if not hasattr(response, "success") or getattr(response, "success") is False:
            raise RuntimeError("Could not start feature generation job.")
        job_id = response.id

        # Wait for job to complete
        try:
            response = poll(
                jobs_client=jobs_api,
                project_id=project_id,
                job_id=job_id,
                timeout_sec=JOB_TIMEOUT,
            )
        except TimeoutException as te:
            raise te
        except Exception as e:
            raise e
        logging.info(response)

        # Define training request
        keras_parameter_request = SetKerasParameterRequest.from_dict(
            {
                "mode": "visual",
                "training_cycles": 10,
                "learning_rate": 0.001,
                "train_test_split": 0.8,
                "skip_embeddings_and_memory": True,
            }
        )

        # Train model
        response = jobs_api.train_keras_job(
            project_id=project_id,
            learn_id=learn_id,
            set_keras_parameter_request=keras_parameter_request,
        )

        if not hasattr(response, "success") or getattr(response, "success") is False:
            raise RuntimeError("Could not start feature generation job.")
        job_id = response.id

        # Wait for job to complete
        try:
            response = poll(
                jobs_client=jobs_api,
                project_id=project_id,
                job_id=job_id,
                timeout_sec=JOB_TIMEOUT,
            )
        except TimeoutException as te:
            raise te
        except Exception as e:
            raise e
        logging.info(response)

        # Test builing/deploying the impulse
        with tempfile.TemporaryDirectory() as dirname:
            model = build(
                deploy_model_type="int8",
                engine="tflite",
                deploy_target="zip",
                output_directory=dirname,
                timeout_sec=JOB_TIMEOUT,
            )
            self.verify_valid_zip_file(self.get_only_file(dirname), model)
