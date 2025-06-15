# eval_openskill.py
# Copyright 2025 Patrick Meade
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import jsonschema
from openskill.models import MODELS

EVAL_SCHEMA_PATH = "schema.json"


def _get_model(name: str) -> Any:
    """Obtain a model object from its class name."""
    for model in MODELS:
        if str(model.__name__) == name:
            return model
    
    raise Exception(f"Unknown model '{name}'")


def _validate_models(openskill_models: List[str]) -> bool:
    """
    Determine if `openskill_models` contains all the models provided by OpenSkill.

    If OpenSkill removes a model, then testing against it will go poorly.
    If OpenSkill adds a new model, it means the Rust implemenation has a coverage gap.

    :return: `True` if the test harness models match the OpenSkill models.
    `False` if a model has been added to or gone missing from OpenSkill.
    """
    all_ok = True

    OPENSKILL_MODEL_NAMES = [str(model.__name__) for model in MODELS]

    for model in OPENSKILL_MODEL_NAMES:
        if model not in openskill_models:
            print(f"OpenSkill may have added a new model: '{model}'")
            all_ok = False

    for model in openskill_models:
        if model not in OPENSKILL_MODEL_NAMES:
            print(f"OpenSkill may have removed the model: '{model}'")
            all_ok = False

    return all_ok


def _validate_params(request: Dict[str, Any]) -> bool:
    """Determine if the `params` provided are appropriate for the `model` specified."""
    # if algorithm parameters were not provided, we're good
    if "params" not in request:
        return True

    # otherwise, we're looking for mis-use of the `window_size` parameter
    # if it's a Part model, it can use `window_size`; we're all good here
    model = request["model"]
    if model in [ "BradleyTerryPart", "ThurstoneMostellerPart" ]:
        return True

    # since it's not a Part model, we need to ensure `window_size` wasn't specified
    params = request["params"]
    if "window_size" in params:
        print(f"'window_size' is not a parameter of the '{model}' model.")
        return False

    # okay, params look good to go
    return True


def _validate_req(request: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Determine if the provided `request` is a valid OpenSkill evaluation request."""
    try:
        jsonschema.validate(request, schema)
    except jsonschema.exceptions.SchemaError as e:
        print(f"OpenSkill evaluation request schema is not a valid JSON schema:\n{e}")
        return False
    except jsonschema.exceptions.ValidationError as e:
        print(f"Provided JSON is an invalid OpenSkill evaluation request:\n{e}")
        return False

    if not _validate_params(request):
        return False

    if not _validate_shape(request):
        return False

    return True


def _validate_shape(request: Dict[str, Any]) -> bool:
    """
    Determine if the `ranks`, `scores`, and `weights` parameters match the shape of
    the `teams` parameter for a `rate` request.
    """
    # bail if this isn't a `rate` request
    if request["func"] != "rate":
        return True

    # these were validated in the schema
    data = request["data"]
    teams = data["teams"]

    # if specified, check that we've got a rank, score, or list of weights for each team
    for check_param in ["ranks", "scores", "weights"]:
        if check_param in data:
            param_value = data[check_param]
            if len(param_value) != len(teams):
                print(f"'{check_param}' must have the same number of elements as 'teams', not {len(param_value)}.")
                return False

    # if `weights` is specified, check that we've got a weight for each player on each team
    if "weights" in data:
        weights = data["weights"]
        for index, team_weights in enumerate(weights):
            if len(team_weights) != len(teams[index]):
                print(f"'weights[{index}]' must have the same number of elements as teams[{index}], not {len(weights[index])}")
                return False

    # okay, the shapes are consistent; carry on
    return True


# def _try_something():
#     # "BradleyTerryFull",
#     # "BradleyTerryPart",
#     # "PlackettLuce",
#     # "ThurstoneMostellerFull",
#     # "ThurstoneMostellerPart"
#     model = _get_model("PlackettLuce")()
#     p1 = model.rating(15.0, 0.000001)
#     print(p1)
#     p2 = model.rating(25.0, 0.000001)
#     print(p2)
#     p3 = model.rating(35.0, 0.000001)
#     print(p3)
#     p4 = model.rating(45.0, 0.000001)
#     print(p4)
#     result = model.predict_draw([[p1, p2], [p3, p4]])
#     print(result)
#     result = model.predict_draw([[p1, p3], [p2, p4]])
#     print(result)
#     result = model.predict_draw([[p1, p4], [p2, p3]])
#     print(result)


def main():
    """Evaluate a test case using OpenSkill.py"""
    schema = json.loads(Path(EVAL_SCHEMA_PATH).read_text())
    openskill_models = schema["properties"]["model"]["enum"]

    if not _validate_models(openskill_models):
        sys.exit(1)

    request = json.load(sys.stdin)
    if not _validate_req(request, schema):
        sys.exit(1)

    # _try_something()

    print("evaluation complete")


if __name__ == "__main__":
    main()
