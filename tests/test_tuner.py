# ruff: noqa: D100, D101, D102, D103
import pickle
import unittest
import logging
import edgeimpulse

from edgeimpulse.datasets import download_dataset
from edgeimpulse.experimental.data import (
    upload_directory,
    infer_category_and_label_from_filename,
)

from edgeimpulse.tuner._functions.tuner import (
    check_tuner,
    print_tuner_coordinator_logs,
    print_tuner_job_logs,
    set_impulse_from_trial,
    start_tuner,
    tuner_report_as_df,
    list_tuner_runs,
    get_tuner_run_state,
)
from edgeimpulse.util import configure_generic_client, default_project_id_for

from edgeimpulse_api import (
    UpdateProjectRequest,
    ProjectsApi,
    ImpulseApi,
    GetJWTRequest,
    Impulse,
    ApiClient,
    LoginApi,
    Configuration,
)

from tests.util import delete_all_samples


logging.getLogger().setLevel(logging.DEBUG)

TUNER_TIMEOUT_SEC = 3600


def set_advanced_tuner_experiment():
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)
    projects_api = ProjectsApi(client)
    x = projects_api.get_project_info(project_id=project_id)

    experiment = "tuner_advanced"

    exps = list(exp.type for exp in x.experiments)
    if experiment in exps:
        return

    exps += [experiment]

    # Let's set the client via the JWT token

    if not edgeimpulse.EI_USERNAME or not edgeimpulse.EI_PASSWORD:
        raise Exception("Either EI_USERNAME or EI_PASSWORD isn't set")

    login_api = LoginApi(client)
    jwt = login_api.login(
        GetJWTRequest(
            username=edgeimpulse.EI_USERNAME, password=edgeimpulse.EI_PASSWORD
        )
    )

    jwt_client = ApiClient(
        Configuration(
            host=edgeimpulse.API_ENDPOINT,
            api_key={"JWTHttpHeaderAuthentication": jwt.token},
        )
    )

    projects_api = ProjectsApi(jwt_client)
    projects_api.update_project(
        project_id=project_id,
        update_project_request=UpdateProjectRequest(experiments=exps),
    )


def set_impulse() -> None:
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)
    impulse_api = ImpulseApi(client)

    impulse = Impulse.from_json(
        """
        {
        "inputBlocks": [
            {
            "id": 222,
            "type": "time-series",
            "name": "Time series",
            "title": "Time series data",
            "windowSizeMs": 2000,
            "windowIncreaseMs": 200,
            "frequencyHz": 62.5,
            "padZeros": true,
            "createdBy": "createImpulse",
            "createdAt": "2024-02-29T15:40:19.667Z"
            }
        ],
        "dspBlocks": [
            {
            "id": 333,
            "type": "spectral-analysis",
            "name": "Spectral features",
            "axes": ["accX", "accY", "accZ"],
            "title": "Spectral Analysis",
            "input": 222,
            "createdBy": "createImpulse",
            "createdAt": "2024-02-29T15:40:19.667Z",
            "implementationVersion": 4
            }
        ],
        "learnBlocks": [
            {
            "id": 7,
            "type": "keras",
            "name": "Classifier",
            "dsp": [333],
            "title": "Classification",
            "createdBy": "createImpulse",
            "createdAt": "2024-02-29T15:40:19.667Z"
            }
        ]
        }

        """
    )
    impulse_api.delete_impulse(project_id=project_id)
    impulse_api.create_impulse(project_id=project_id, impulse=impulse)


def upload_gestures():
    upload_directory(
        directory="datasets/gestures",
        transform=infer_category_and_label_from_filename,
    )


def load_state(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class TestTuner(unittest.TestCase):
    def setUp(self):
        delete_all_samples()

    def test_tuner(self):
        set_advanced_tuner_experiment()

        download_dataset("gestures")
        upload_gestures()

        set_impulse()

        start_tuner(
            name="Simple classification",
            target_device="jetson-nano",
            classification_type="classification",
            dataset_category="motion_continuous",
            target_latency=1,
            tuning_max_trials=3,
        )

        state = None

        try:
            state = check_tuner(timeout_sec=TUNER_TIMEOUT_SEC, wait_for_completion=True)
            print(state)
        except Exception as e:
            print("An error occurred:", e)

        print("--------------------------")
        print("COORDINATOR LOGS")
        print("--------------------------")

        print_tuner_coordinator_logs()

        print("--------------------------")
        print("JOB LOGS")
        print("--------------------------")

        print_tuner_job_logs()

        print("--------------------------")
        print("REPORT AS DF")
        print("--------------------------")

        df = tuner_report_as_df(state)

        print("--------------------------")
        print(df)
        print("--------------------------")
        df = df.sort_values(by="val_float32_accuracy", ascending=False)

        trial_id = df.iloc[0].trial_id
        set_impulse_from_trial(trial_id=trial_id)

        res = list_tuner_runs()
        tuner_coordinator_job_id = res.runs[0].tuner_coordinator_job_id
        state = get_tuner_run_state(tuner_coordinator_job_id=tuner_coordinator_job_id)
        df = tuner_report_as_df(state)

        print(df)
