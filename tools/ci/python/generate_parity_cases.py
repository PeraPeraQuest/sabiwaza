# generate_parity_cases.py
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

import random
import json
from pathlib import Path

from eval_openskill import eval_request

def random_rating():
    return {
        "mu": round(random.uniform(10, 40), 6),
        "sigma": round(random.uniform(2, 12), 6)
    }

def random_team(min_players=1, max_players=4):
    return [random_rating() for _ in range(random.randint(min_players, max_players))]

def random_teams(min_teams=2, max_teams=4):
    return [random_team() for _ in range(random.randint(min_teams, max_teams))]

def random_weights(teams):
    return [[round(random.uniform(0.5, 2.0), 6) for _ in team] for team in teams]

def generate_params():
    return {
        "mu": round(random.uniform(20.0, 30.0), 6),
        "sigma": round(random.uniform(6.0, 10.0), 6),
        "beta": round(random.uniform(2.0, 6.0), 6),
        "kappa": 0.0001,
        "tau": round(random.uniform(0.05, 0.15), 6),
        "margin": round(random.uniform(0.0, 5.0), 6),
        "limit_sigma": random.choice([True, False]),
        "balance": random.choice([True, False])
    }

def make_case(index: int):
    teams = random_teams()
    n_teams = len(teams)

    ranks_or_scores = random.choice(["ranks", "scores", None])
    ranks = list(range(n_teams))
    random.shuffle(ranks)
    scores = [round(random.uniform(0, 100), 6) for _ in range(n_teams)]

    weights = random_weights(teams)

    data = {
        "teams": teams,
        "weights": weights
    }

    if ranks_or_scores == "ranks":
        data["ranks"] = ranks
    elif ranks_or_scores == "scores":
        data["scores"] = scores

    if random.choice([True, False]):
        data["tau"] = round(random.uniform(0.01, 0.2), 6)
    if random.choice([True, False]):
        data["limit_sigma"] = random.choice([True, False])

    return {
        "model": "BradleyTerryFull",
        "func": "rate",
        "params": generate_params(),
        "data": data
    }

def eval_openskill_result(case):
    return eval_request(case)

# Write a batch of test cases to a folder
def write_cases(path: Path, count: int = 10):
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        case = make_case(i)
        case["result"] = eval_openskill_result(case)
        with open(path / f"rate_{i:03}.json", "w") as f:
            json.dump(case, f, indent=4)

# Example usage
if __name__ == "__main__":
    write_cases(Path("tests/parity_cases"), count=5)
