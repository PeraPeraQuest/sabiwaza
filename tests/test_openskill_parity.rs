// test_openskill_parity.rs
// Copyright 2025 Patrick Meade
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::fs;

use serde_json::{Value, json};

use sabiwaza::{algorithm::SkillAlgorithm, rating::SkillRating};

#[test]
fn test_always_succeed() {
    assert!(true);
}

/// Run all parity test cases in tests/parity_cases
#[test]
fn run_all_parity_cases() {
    let cases = load_test_cases("tests/parity_cases");

    for (filename, request) in cases {
        let rust_result = run_rust(&request);
        let python_result = run_python(&request);
        let expected_result = request.get("result");

        // If there's a locked-in expected result, validate both against it
        if let Some(expected) = expected_result {
            assert!(
                json_approx_equal(expected, &python_result, 1e-6),
                "FAIL in {}: OpenSkill Python output drifted:\nexpected = {:#?}\nactual   = {:#?}",
                filename,
                expected,
                python_result
            );
            assert!(
                json_approx_equal(expected, &rust_result, 1e-6),
                "FAIL in {}: Rust output mismatched locked-in result:\nexpected = {:#?}\nactual   = {:#?}",
                filename,
                expected,
                rust_result
            );
        } else {
            // Fall back to comparing rust ↔ python
            assert!(
                json_approx_equal(&rust_result, &python_result, 1e-6),
                "FAIL in {}: rust = {:#?}, python = {:#?}",
                filename,
                rust_result,
                python_result
            );
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------

fn json_approx_equal(a: &Value, b: &Value, epsilon: f64) -> bool {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64().unwrap();
            let fb = nb.as_f64().unwrap();
            (fa - fb).abs() < epsilon
        }
        (Value::Array(arr_a), Value::Array(arr_b)) => {
            if arr_a.len() != arr_b.len() {
                return false;
            }
            arr_a
                .iter()
                .zip(arr_b.iter())
                .all(|(av, bv)| json_approx_equal(av, bv, epsilon))
        }
        (Value::Object(map_a), Value::Object(map_b)) => {
            if map_a.len() != map_b.len() {
                return false;
            }
            map_a.iter().all(|(k, va)| match map_b.get(k) {
                Some(vb) => json_approx_equal(va, vb, epsilon),
                None => false,
            })
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => false, // mismatched types or unsupported types
    }
}

fn load_test_cases(dir: &str) -> Vec<(String, Value)> {
    let mut cases = vec![];

    for entry in fs::read_dir(dir).expect("test case dir not found") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let contents = fs::read_to_string(&path).unwrap();
            let json: Value = serde_json::from_str(&contents).unwrap();
            let filename = path.file_name().unwrap().to_string_lossy().to_string();
            cases.push((filename, json));
        }
    }

    cases
}

fn new_model(params: &Value) -> SkillAlgorithm {
    fn get_f64(params: &Value, key: &str, default: f64) -> f64 {
        params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    fn get_bool(params: &Value, key: &str, default: bool) -> bool {
        params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    // These are the same default values used in SkillAlgorithm::default()
    let mu = get_f64(params, "mu", 25.0);
    let sigma = get_f64(params, "sigma", 25.0 / 3.0);
    let beta = get_f64(params, "beta", 25.0 / 6.0);
    let kappa = get_f64(params, "kappa", 0.0001);
    let tau = get_f64(params, "tau", 25.0 / 300.0);
    let margin = get_f64(params, "margin", 0.0);
    let limit_sigma = get_bool(params, "limit_sigma", false);
    let balance = get_bool(params, "balance", false);

    SkillAlgorithm {
        mu,
        sigma,
        beta,
        kappa,
        tau,
        margin,
        limit_sigma,
        balance,
    }
}

fn run_python(request: &Value) -> Value {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let input = serde_json::to_string(request).unwrap();
    let mut child = Command::new("python3")
        .arg("tools/ci/python/eval_openskill.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to run Python");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();

    if !output.status.success() {
        panic!("Python error: {}", String::from_utf8_lossy(&output.stdout));
    }

    let v: Value = serde_json::from_slice(&output.stdout).unwrap();
    v.get("result").cloned().expect("missing 'result'")
}

fn run_rust(request: &Value) -> Value {
    let model_name = request.get("model").expect("missing model");
    if model_name != "BradleyTerryFull" {
        panic!("sabiwaza does not support model '{model_name}'");
    }
    let model = match request.get("params") {
        Some(params) => new_model(params),
        None => SkillAlgorithm::default(),
    };
    let func = request
        .get("func")
        .expect("missing func")
        .as_str()
        .expect("non-string func");
    let data = request.get("data").expect("missing data");

    let result = match func {
        "ordinal" => run_rust_ordinal(&model, data),
        "rate" => run_rust_rate(&model, data),
        _ => {
            panic!("unknown func '{func}'")
        }
    };

    result
}

fn run_rust_ordinal(model: &SkillAlgorithm, data: &Value) -> Value {
    fn get_f64(params: &Value, key: &str, default: f64) -> f64 {
        params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    let z = get_f64(data, "z", 3.0);
    let alpha = get_f64(data, "alpha", 1.0);
    let target = get_f64(data, "target", 0.0);

    let default_rating_data = serde_json::json!({});
    let rating_data = data.get("rating").unwrap_or(&default_rating_data);
    let mu = rating_data
        .get("mu")
        .and_then(|v| v.as_f64())
        .unwrap_or(model.mu);
    let sigma = rating_data
        .get("sigma")
        .and_then(|v| v.as_f64())
        .unwrap_or(model.sigma);

    let skill_rating = SkillRating::new(mu, sigma);
    let ordinal = skill_rating.ordinal(z, alpha, target);

    json!(ordinal)
}

fn run_rust_rate(model: &SkillAlgorithm, data: &Value) -> Value {
    // Parse required teams
    let teams_json = data.get("teams").expect("Missing 'teams' field");
    let teams: Vec<Vec<SkillRating>> = teams_json
        .as_array()
        .expect("'teams' should be an array of arrays")
        .iter()
        .map(|team_json| {
            team_json
                .as_array()
                .expect("Each team should be an array of players")
                .iter()
                .map(|player_json| {
                    let mu = player_json
                        .get("mu")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(model.mu);
                    let sigma = player_json
                        .get("sigma")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(model.sigma);
                    SkillRating::new(mu, sigma)
                })
                .collect()
        })
        .collect();

    // Optional overrides
    let tau = data.get("tau").and_then(|v| v.as_f64());
    let limit_sigma = data.get("limit_sigma").and_then(|v| v.as_bool());

    // Optional ranks
    let ranks: Option<Vec<f64>> = data.get("ranks").and_then(|v| {
        v.as_array()
            .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
    });

    // Optional scores
    let scores: Option<Vec<f64>> = data.get("scores").and_then(|v| {
        v.as_array()
            .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
    });

    // Optional weights
    let weights: Option<Vec<Vec<f64>>> = data.get("weights").and_then(|outer| {
        outer.as_array().map(|teams| {
            teams
                .iter()
                .map(|team| {
                    team.as_array()
                        .map(|players| players.iter().filter_map(|x| x.as_f64()).collect())
                        .unwrap_or_default()
                })
                .collect()
        })
    });

    // Create a copy of the model with any tau/limit_sigma overrides
    let mut model = model.clone();
    if let Some(t) = tau {
        model.tau = t;
    }
    if let Some(ls) = limit_sigma {
        model.limit_sigma = ls;
    }

    // Compute results
    let new_teams = model
        .rate(&teams, ranks, scores, weights)
        .expect("expected rate");

    // Format as JSON
    let json_result = json!(
        new_teams
            .iter()
            .map(|team| {
                team.iter()
                    .map(|player| {
                        json!({
                            "mu": player.mu,
                            "sigma": player.sigma
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    );

    json_result
}
