"""Use this module to work with the Edge Impulse EON tuner."""

from copy import deepcopy
import edgeimpulse
import time
import re
import logging

from edgeimpulse.util import (
    configure_generic_client,
    default_project_id_for,
    pandas_installed,
    poll,
)

from edgeimpulse_api import (
    OptimizeConfig,
    OptimizationApi,
    JobsApi,
    OptimizeStateResponse,
    TunerTrial,
    StartJobResponse,
    ListTunerRunsResponse,
    TunerSpaceImpulse,
    OptimizeConfigSearchSpaceTemplate,
)

from edgeimpulse.exceptions import (
    TimeoutException,
)

from typing import Any, Dict, List, Optional, Union


def list_tuner_runs() -> ListTunerRunsResponse:
    """List the tuner runs that have been done in the current project.

    Returns:
        ListTunerRunsResponse: An object containing all the tuner runs
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)
    result = optimize_api.list_tuner_runs(project_id=project_id)
    return result


def print_tuner_coordinator_logs(limit: int = 500) -> None:
    """Retrieve and print logs for the tuner coordinator job.

    Returns:
        None
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)
    state = optimize_api.get_state(project_id)

    try:
        state = optimize_api.get_state(project_id)
    except Exception as e:
        logging.debug(f"Failed getting state  [{str(e)}]")
        raise e

    if state.tuner_coordinator_job_id is None:
        raise Exception("Can't find coordinator job id, has the coordinator started?")

    print_job_logs(state.tuner_coordinator_job_id, limit=limit)


def print_tuner_job_logs(limit: int = 500) -> None:
    """Retrieve and print logs for the tuner job.

    Returns:
        None
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)

    try:
        state = optimize_api.get_state(project_id)
    except Exception as e:
        logging.debug(f"Failed getting state  [{str(e)}]")
        raise e

    if state.tuner_job_id is None:
        raise Exception("Can't find tuner job in the state, has it started?")

    print_job_logs(state.tuner_job_id, limit=limit)


def print_job_logs(job_id: str, limit: int = 500) -> None:
    """Retrieve and print the logs for a specific job.

    Args:
        job_id (str): The ID of the job for which to retrieve logs.
        limit (int):  Limit of logs to retrieve. Default is 500

    Returns:
        None
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    jobs_api = JobsApi(client)
    project_id = default_project_id_for(client)

    logs = jobs_api.get_jobs_logs(project_id, job_id, limit=limit)

    [
        print(f"[{logs.created}] {(logs.log_level or '').upper()}: {logs.data}")
        for logs in reversed(logs.stdout)
    ]


def get_tuner_url(project_id: int, tuner_coordinator_job_id: int) -> str:
    """Generate the URL to view tuner results for a specific project and tuner job.

    Args:
        project_id (int): The ID of the project.
        tuner_coordinator_job_id (int): The ID of the tuner coordinator job (see the `state` object)

    Returns:
        str: The URL to view the tuner results.
    """
    return f"{edgeimpulse.API_ENDPOINT.replace('/v1', '')}/studio/{project_id}/tuner/{tuner_coordinator_job_id} to view results"


def start_tuner(
    space: List[TunerSpaceImpulse],
    target_device: str,
    target_latency: int,
    tuning_max_trials: Optional[int] = None,
    name: Optional[str] = None,
) -> StartJobResponse:
    """Start the EON tuner with default settings. Use `start_custom_tuner` to specify config.

    Args:
        space (List[TunerSpaceImpulse]): The search space for the tuner.
        target_device (str): The target device for optimization. Use `get_profile_devices() to
            get the the available devices.
        target_latency (int): The target latency for the model in ms.
        tuning_max_trials (int, optional): The maximum number of tuning trials.
            None means let tuner decide. Defaults to None.
        name (str, optional): Name to give this run. Default is None.

    Returns:
        StartJobResponse: The response containing information about the started job.
    """
    config = OptimizeConfig(
        name=name,
        space=space,
        target_device={"name": target_device},
        target_latency=target_latency,
        tuning_max_trials=tuning_max_trials,
    )
    return start_custom_tuner(config)


def start_tuner_template(
    template: OptimizeConfigSearchSpaceTemplate,
    target_device: str,
    target_latency: int,
    tuning_max_trials: Optional[int] = None,
    name: Optional[str] = None,
) -> StartJobResponse:
    """Start the EON tuner with default settings. Use `start_custom_tuner` to specify config.

    Args:
        template (OptimizeConfigSearchSpaceTemplate): The search space template for the tuner.
        target_device (str): The target device for optimization. Use `get_profile_devices() to
            get the the available devices.
        target_latency (int): The target latency for the model in ms.
        tuning_max_trials (int, optional): The maximum number of tuning trials.
            None means let tuner decide. Defaults to None.
        name (str, optional): Name to give this run. Default is None.

    Returns:
        StartJobResponse: The response containing information about the started job.
    """
    config = OptimizeConfig(
        name=name,
        search_space_template=template,
        target_device={"name": target_device},
        target_latency=target_latency,
        tuning_max_trials=tuning_max_trials,
    )
    return start_custom_tuner(config)


def start_custom_tuner(config: OptimizeConfig) -> StartJobResponse:
    """Start a tuner job with custom configuration.

    Args:
        config (OptimizeConfig): The custom configuration for the tuner job.

    Returns:
        StartJobResponse: The response object indicating the status of the job.
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )

    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)

    try:
        res = res = optimize_api.update_config(project_id, config)
    except Exception as e:
        logging.debug(
            f"Failed setting the config of the tuner while running start tuner [{str(e)}]"
        )
        raise e

    jobs_api = JobsApi(client)

    try:
        res = jobs_api.optimize_job(project_id)
    except Exception as e:
        logging.debug(f"Failed starting the tuner job [{str(e)}]")
        raise e

    if not res or not res.id:
        raise ValueError("Failed to fetch the job id from the optimize job")

    print(f"Starting tuner {get_tuner_url(project_id, res.id)} to view results")
    return res


def check_tuner(
    timeout_sec: Optional[int] = None, wait_for_completion: bool = True
) -> OptimizeStateResponse:
    """Check the current state of the tuner and optionally waits until the tuner has completed.

    Args:
        timeout_sec (int, optional): The maximum time to wait for the tuner to start trials, in seconds. Defaults to None.
        wait_for_completion (bool, optional): If True, waits for the tuner to complete; if False, returns immediately
            after checking the tuner state. Defaults to True.

    Returns:
        OptimizeStateResponse: The current state of the tuner.

    Raises:
        Exception: If the timeout_sec is reached while waiting for trials to start.

    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    optimize_api = OptimizationApi(client)
    project_id = default_project_id_for(client)
    state = optimize_api.get_state(project_id)  ##TODO: Add guard here
    print_tuner(state)

    while not state.trials:
        if state.status.status == "completed":
            if state.job_error:
                print(f"Job is completed with error: {state.job_error}")
                print_job_logs(state.tuner_job_id)
                return state
            else:
                if len(state.trials) == 0:
                    print(
                        "Job completed but no trials found, check tuner job logs or tuner coordinator logs `get_tuner_coordinator_logs()`."
                    )
                elif any(trial.status != "completed" for trial in state.trials):
                    print(
                        "Job completed but some trials aren't in the completed, check the trial logs."
                    )
                else:
                    print("Job already completed")
                return state
            return

        print(f"Waiting for trials to start for project id: {project_id}")
        time.sleep(5)
        state = optimize_api.get_state(project_id)  ## add guard here
        print_tuner(state)
        if isinstance(timeout_sec, int):
            timeout_sec -= 5
            if timeout_sec <= 0:
                raise Exception(
                    f"Timeout waiting for trials to start for project id: {project_id}. Check the coordinator logs  "
                    + "via get_tuner_coordinator_logs()`. Usually it means that we can't find a model close to your latency, "
                    + "try to make your latency smaller."
                )

    tuner_job_id = state.tuner_job_id
    if tuner_job_id is None:
        print(f"No tuner job found for project id: {project_id}")
    else:
        state = optimize_api.get_state(project_id)
        while not state.status.status == "completed":
            state = optimize_api.get_state(project_id)  ## add guard here
            print_tuner(state)
            if not wait_for_completion:
                print("EON tuner jobs still running. Check back later")
                return
            time.sleep(5)
        print(f"Tuner job in project id: {project_id} finished")
        print_tuner(state)
        return state


def get_tuner_run_state(tuner_coordinator_job_id: int) -> OptimizeStateResponse:
    """Retrieve the current state of the tuner run.

    Returns:
        OptimizeStateResponse: The OptimizeStateResponse object representing the current Tuner state.
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)
    state = optimize_api.get_tuner_run_state(
        project_id=project_id, tuner_coordinator_job_id=tuner_coordinator_job_id
    )
    return state


def get_tuner_state() -> OptimizeStateResponse:
    """Retrieve the current state of the tuner run.

    Returns:
        OptimizeStateResponse: The OptimizeStateResponse object representing the current Tuner state.
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY,
        host=edgeimpulse.API_ENDPOINT,
    )
    project_id = default_project_id_for(client)
    optimize_api = OptimizationApi(client)
    state = optimize_api.get_state(project_id)
    return state


def set_impulse_from_trial(
    trial_id: str,
    timeout_sec: Optional[float] = None,
    wait_for_completion: Optional[bool] = True,
) -> StartJobResponse:
    """Replace the current Impulse configuration with one found in a trial fromm the tuner.

    Args:
        trial_id (string): The trial id
        timeout_sec (float, optional): The maximum time to wait for the tuner to complete, in seconds. Defaults to None.
        wait_for_completion (bool, optional): If True, waits for the tuner to complete; if False, returns immediately

    Returns:
        StartJobResponse: The response object indicating the status of the job.
    """
    client = configure_generic_client(
        key=edgeimpulse.API_KEY, host=edgeimpulse.API_ENDPOINT
    )
    project_id = default_project_id_for(client)
    jobs_api = JobsApi(client)
    res = jobs_api.set_tuner_primary_job(project_id=project_id, trial_id=trial_id)
    if not res or not res.success:
        raise ValueError("Failed set current impulse", res.error)

    # Wait for the job to complete
    if wait_for_completion:
        job_id = res.id
        try:
            job_response = poll(
                jobs_client=jobs_api,
                project_id=project_id,
                job_id=job_id,
                timeout_sec=timeout_sec,
            )
        except TimeoutException as te:
            raise te
        except Exception as e:
            raise e
        logging.info(job_response)

    return res


def get_trial_parameters(trial: TunerTrial) -> dict:
    """Retrieve flattened parameters for the input, dsp, and learn blocks from a TunerTrialImpulse.

    Args:
        trial (TunerTrialImpulse): The TunerTrialImpulse object containing the model parameters.

    Returns:
        dict: A dictionary containing all flattened parameters for input, dsp, and learn blocks.
    """
    flatten = _flatten_dict(trial.model)
    return flatten


def get_trial_metrics(trial: TunerTrial) -> dict:
    """Retrieve flattened metrics from the learn block for various precisions (e.g., float32, int8) for both test and validation.

    Args:
        trial (TunerTrialImpulse): The TunerTrialImpulse object representing the trial.

    Returns:
        dict: A dictionary containing flattened metrics including validation and test accuracy, loss, size, and memory usage.
    """
    if trial.status != "completed":
        return {}
    else:
        block_index = 0  # TODO: for now only support one learn-block
        block = trial.impulse.learn_blocks[block_index]
        metrics_list = deepcopy(block["metadata"]["modelValidationMetrics"])
        types = [metric.pop("type") for metric in metrics_list]

        val_precisions = [f"val_{t}" for t in types]
        new_data = {}

        keep = ["loss", "size", "memory", "accuracy"]
        for t, old_dict in zip(val_precisions, metrics_list):
            for k in keep:
                new_data[f"{t}_{k}"] = old_dict[k]

        flatten = _flatten_dict(new_data)
        flatten = {**flatten}

        for precision in types:
            flatten[f"test_{precision}_accuracy"] = block["metrics"]["test"][precision][
                "accuracy"
            ]
        return flatten


def print_tuner(state: OptimizeStateResponse) -> None:
    """Print the state of the tuner worker and trials.

    Args:
        state (OptimizeStateResponse): The OptimizeStateResponse object representing the current Tuner state.

    Returns:
        None
    """
    status = state.status

    print(10 * "--")
    print(
        f"- Tuner run  | #{state.tuner_job_id or '(Waiting for job)'} {state.config.name or '(No run name)'} ({state.status.status}) - Target: {state.config.target_device.name}"
    )
    print(
        f"- Workers    | Ready: {status.num_ready_workers} Busy: {status.num_busy_workers} Pending: {status.num_pending_workers}"
    )
    print(
        f"- Trials     | Running: {status.num_running_trials} Completed: {status.num_completed_trials} Pending: {status.num_pending_trials} Failed: {status.num_failed_trials}"
    )

    print(10 * "--")


def tuner_report_as_df(state: OptimizeStateResponse):
    """Get a tuner trial report dataframe with model metrics and block configuration.

    This method needs pandas to be installed.

    Generate a dataframe on the tuner trials including used input, model, learn block configuration and model
    validation metrics.

    Args:
        state (OptimizeStateResponse): The tuner state containing tuner trials.

    Returns:
        pd.DataFrame: A DataFrame containing all model parameters and validation metrics.
    """
    if not pandas_installed():
        raise Exception("Please install pandas to use this function")

    import pandas as pd

    def get_trial_standard_info(trial: TunerTrial):
        return {
            "trial_id": trial.id,
            "status": trial.status,
            "last_completed_training": trial.last_completed_training,
        }

    metrics = [
        {
            **get_trial_standard_info(trial),
            **get_trial_parameters(trial),
            **get_trial_metrics(trial),
            **_flatten_dict({"devicePerformance": trial.device_performance}),
        }
        for trial in state.trials
    ]

    return pd.DataFrame(_underscore_keys(metrics))


def _to_underscore(key: str) -> str:
    """Convert a camelCase string to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])|[^a-zA-Z0-9]", "_", key).lower()


def _underscore_keys(obj: Union[Dict[Any, Any], List[Any]]) -> Union[dict, List[Any]]:
    """Convert keys of a dictionary from camelCase to snake_case in a recursive manner."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                value = _underscore_keys(value)
            new_dict[_to_underscore(key)] = value
        return new_dict
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            if isinstance(item, (dict, list)):
                item = _underscore_keys(item)
            new_list.append(item)
        return new_list
    else:
        return obj


def _flatten_dict(item, pk=""):
    """Recursively flattens a dictionary."""
    items = []
    if isinstance(item, dict):
        values = item.items()
    elif isinstance(item, list):
        values = enumerate(item)
    else:
        raise Exception("item should be a dict or a list")

    for index, value in values:
        nk = pk + "_" + str(index) if pk != "" else str(index)
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, pk=nk).items())
        elif isinstance(value, list):
            items.extend(_flatten_dict(value, pk=nk).items())
        else:
            items.append([nk, value])

    return dict(items)
