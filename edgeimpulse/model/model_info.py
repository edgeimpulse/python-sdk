# ruff: noqa: D100, D101, D107

from typing import Optional

from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_input import (
    DeployPretrainedModelRequestModelInfoInput,
)
from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_model import (
    DeployPretrainedModelRequestModelInfoModel,
)

from edgeimpulse.model.input_type import OtherInput


# Parameter name `input` is OK to shadow Python builtin
# ruff: noqa: A002 A001
class ModelInfo(dict):
    def __init__(self, model: dict, input: Optional[dict] = None):
        if input is None:
            input = OtherInput()
        self["input"] = DeployPretrainedModelRequestModelInfoInput.from_dict(input)
        self["model"] = DeployPretrainedModelRequestModelInfoModel.from_dict(model)
