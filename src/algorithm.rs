// algorithm.rs
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

use std::collections::HashMap;

use crate::error::{Result, SabiwazaError};
use crate::rating::SkillRating;

/// The BradleyTerryFull model assumes a single scalar value to
/// represent player performance, allows for rating updates based on match
/// outcomes, and uses a logistic regression approach to estimate player
/// ratings.
#[derive(Clone)]
pub struct SkillAlgorithm {
    /// Represents the initial belief about the skill of
    /// a player before any matches have been played. Known
    /// mostly as the mean of the Guassian prior distribution.
    pub mu: f64,

    /// Standard deviation of the prior distribution of player.
    pub sigma: f64,

    /// Hyperparameter that determines the level of uncertainty
    /// or variability present in the prior distribution of
    /// ratings.
    pub beta: f64,

    /// Arbitrary small positive real number that is used to
    /// prevent the variance of the posterior distribution from
    /// becoming too small or negative. It can also be thought
    /// of as a regularization parameter.
    pub kappa: f64,

    /// Additive dynamics parameter that prevents sigma from
    /// getting too small to increase rating change volatility.
    pub tau: f64,

    /// The margin of victory needed for a win to be considered
    /// impressive.
    pub margin: f64,

    /// Boolean that determines whether to restrict
    /// the value of sigma from increasing.
    pub limit_sigma: bool,

    /// Boolean that determines whether to emphasize
    /// rating outliers.
    pub balance: bool,
}

impl Default for SkillAlgorithm {
    fn default() -> Self {
        Self {
            mu: 25.0,
            sigma: 25.0 / 3.0,
            beta: 25.0 / 6.0,
            kappa: 0.0001,
            tau: 25.0 / 300.0,
            margin: 0.0,
            limit_sigma: false,
            balance: false,
        }
    }
}

impl SkillAlgorithm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mu: f64,
        sigma: f64,
        beta: f64,
        kappa: f64,
        tau: f64,
        margin: f64,
        limit_sigma: bool,
        balance: bool,
    ) -> Self {
        Self {
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
}

impl SkillAlgorithm {
    pub fn rate(
        &self,
        teams: &[Vec<SkillRating>],
        ranks: Option<Vec<f64>>,
        scores: Option<Vec<f64>>,
        weights: Option<Vec<Vec<f64>>>,
    ) -> Result<Vec<Vec<SkillRating>>> {
        // verify there are at least two teams, of at least one player each
        _check_teams(teams)?;
        // verify ranks and scores, if provided
        _check_ranks_and_scores(&ranks, &scores, teams)?;
        // verify weights, if provided
        _check_weights(&weights, teams)?;

        // deep copy teams
        let original_teams = teams;
        let mut teams: Vec<Vec<SkillRating>> = original_teams.to_vec();

        // correct sigma with tau
        let tau_squared = self.tau * self.tau;
        for team in teams.iter_mut() {
            for player in team.iter_mut() {
                player.sigma = (player.sigma * player.sigma + tau_squared).sqrt();
            }
        }

        // convert scores to ranks
        let mut ranks = ranks;
        if ranks.is_none() && scores.is_some() {
            ranks = _convert_scores_to_ranks(&teams, &scores);
        }

        // normalize Weights
        let mut weights = weights;
        if weights.is_some() {
            let some_weights = weights.unwrap();
            let normalized_weights = some_weights
                .iter()
                .map(|team_weights| _normalize(team_weights, 1.0, 2.0))
                .collect();
            weights = Some(normalized_weights);
        }

        // unwind stuff that needs to be unwound (??)
        let mut tenet = None;
        if let Some(some_ranks) = ranks.take() {
            // Unwind teams by rank
            let (ordered_teams, tenet_vec) = _unwind(&some_ranks, &teams);
            teams = ordered_teams;
            tenet = Some(tenet_vec);

            // Unwind weights if present
            if let Some(some_weights) = weights.take() {
                let (ordered_weights, _) = _unwind(&some_ranks, &some_weights);
                weights = Some(ordered_weights);
            }

            // Sort the ranks
            let mut sorted_ranks = some_ranks.clone();
            sorted_ranks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ranks = Some(sorted_ranks);
        }

        // compute some results
        let result = _compute(
            &teams,
            ranks.as_deref(),
            scores.as_deref(),
            weights.as_ref(),
            self.beta,
            self.margin,
            self.kappa,
        );

        // If we need to unwind result to match original sort order
        let processed_result = if let Some(ref original_tenet) = tenet {
            let (unwound_result, _) = _unwind(original_tenet, &result);
            unwound_result
        } else {
            result
        };

        // Possible Final Result
        let final_result = match self.limit_sigma {
            false => processed_result,
            true => {
                let mut limited_result = Vec::new();

                // Reuse processed_result
                for (team_index, team) in processed_result.iter().enumerate() {
                    let mut limited_team = Vec::new();

                    for (player_index, player) in team.iter().enumerate() {
                        let skill_rating = SkillRating {
                            mu: player.mu,
                            sigma: player
                                .sigma
                                .min(original_teams[team_index][player_index].sigma),
                        };
                        limited_team.push(skill_rating);
                    }
                    limited_result.push(limited_team);
                }
                limited_result
            }
        };

        // return the new skill ratings to the caller
        Ok(final_result)
    }
}

impl SkillAlgorithm {
    pub fn predict_win(&self, _teams: &[Vec<SkillRating>]) -> Vec<f64> {
        todo!("Implement BT Full win prediction")
    }

    pub fn predict_draw(&self, _teams: &[Vec<SkillRating>]) -> f64 {
        todo!("Implement BT Full draw prediction")
    }

    pub fn predict_rank(&self, _teams: &[Vec<SkillRating>]) -> Vec<(usize, f64)> {
        todo!("Implement BT Full rank prediction")
    }
}

/// Calculates the rankings based on the scores or ranks of the teams.
///
/// It assigns a rank to each team based on their score, with the team with
/// the highest score being ranked first. Ties are broken by a team's prior
/// averaged mu values.
fn _calculate_rankings(teams: &[Vec<SkillRating>], ranks: Vec<f64>) -> Vec<f64> {
    if teams.is_empty() {
        return Vec::new();
    }

    // If no ranks are provided, return 0..N as default ranks
    if ranks.is_empty() {
        return (0..teams.len()).map(|i| i as f64).collect();
    }

    // Gather (input_rank, avg_ordinal, team_index) for sorting
    let mut team_data: Vec<(f64, f64, usize)> = ranks
        .iter()
        .enumerate()
        .map(|(index, &rank)| {
            let team = &teams[index];
            let avg_ordinal = team
                .iter()
                .map(|player| player.ordinal(3.0, 1.0, 0.0)) // Use default ordinal params
                .sum::<f64>()
                / team.len() as f64;
            (rank, avg_ordinal, index)
        })
        .collect();

    // Sort by rank ascending, then average_ordinal descending
    team_data.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut final_ranks = vec![0.0; teams.len()];
    let mut current_rank = 0.0;

    for (team_index, (orig_rank, avg_ord, orig_index)) in team_data.iter().enumerate() {
        if team_index > 0 {
            let (prev_rank, prev_ord, _) = team_data[team_index - 1];
            if (orig_rank != &prev_rank) || (avg_ord != &prev_ord) {
                current_rank = *orig_index as f64;
            }
        }
        final_ranks[*orig_index] = current_rank;
    }

    final_ranks
}

/// Ensure ranks and scores arguments are valid.
fn _check_ranks_and_scores(
    ranks: &Option<Vec<f64>>,
    scores: &Option<Vec<f64>>,
    teams: &[Vec<SkillRating>],
) -> Result<()> {
    // either ranks or scores, but not both
    if ranks.is_some() && scores.is_some() {
        let msg = "Cannot accept both 'ranks' and 'scores' arguments at the same time.".to_string();
        return Err(SabiwazaError::ValueError(msg));
    }
    // ranks for the rank god
    if ranks.is_some() {
        let num_ranks = ranks.as_ref().unwrap().len();
        if num_ranks != teams.len() {
            let msg = format!(
                "Argument 'ranks' must have the same number of elements as 'teams' not {num_ranks}"
            );
            return Err(SabiwazaError::ValueError(msg));
        }
    }
    // scores for the score god
    if scores.is_some() {
        let num_scores = scores.as_ref().unwrap().len();
        if num_scores != teams.len() {
            let msg = format!(
                "Argument 'scores' must have the same number of elements as 'teams' not {num_scores}"
            );
            return Err(SabiwazaError::ValueError(msg));
        }
    }
    // yep, good to go
    Ok(())
}

/// Ensure teams argument is valid.
fn _check_teams(teams: &[Vec<SkillRating>]) -> Result<()> {
    // make sure we've got at least two teams
    let num_teams = teams.len();
    if num_teams < 2 {
        let msg = format!("Argument 'teams' must have at least 2 teams, not {num_teams}.");
        return Err(SabiwazaError::ValueError(msg));
    }
    // make sure each team has at least one player
    for team in teams {
        let num_players = team.len();
        if num_players < 1 {
            let msg = format!(
                "Argument 'teams' must have at least 1 player per team, not {num_players}."
            );
            return Err(SabiwazaError::ValueError(msg));
        }
    }
    // yep, good to go
    Ok(())
}

/// Ensure weights argument is valid.
fn _check_weights(weights: &Option<Vec<Vec<f64>>>, teams: &[Vec<SkillRating>]) -> Result<()> {
    if weights.is_some() {
        let num_weights = weights.as_ref().unwrap().len();
        if num_weights != teams.len() {
            let msg = format!(
                "Argument 'weights' must have the same number of elements as 'teams', not {num_weights}."
            );
            return Err(SabiwazaError::ValueError(msg));
        }
        for (index, weight) in weights.iter().enumerate() {
            let num_weights = weight.len();
            if num_weights != teams[index].len() {
                let msg = format!(
                    "Argument 'weights' must have the same number of elements as each team in 'teams', not {num_weights}."
                );
                return Err(SabiwazaError::ValueError(msg));
            }
        }
    }
    // yep, good to go
    Ok(())
}

pub fn _compute(
    teams: &[Vec<SkillRating>],
    ranks: Option<&[f64]>,
    scores: Option<&[f64]>,
    weights: Option<&Vec<Vec<f64>>>,
    beta: f64,
    margin: f64,
    kappa: f64,
) -> Vec<Vec<SkillRating>> {
    let mut result: Vec<Vec<SkillRating>> = Vec::with_capacity(teams.len());

    // Compute team-level mu, sigma², and rank
    struct TeamInfo {
        mu: f64,
        sigma_squared: f64,
        rank: usize,
        team: Vec<SkillRating>,
    }

    let mut team_infos: Vec<TeamInfo> = Vec::with_capacity(teams.len());
    for (i, team) in teams.iter().enumerate() {
        let mu: f64 = team.iter().map(|p| p.mu).sum();
        let sigma_squared: f64 = team.iter().map(|p| p.sigma.powi(2)).sum();
        let rank = ranks.map_or(i, |r| r[i] as usize);
        team_infos.push(TeamInfo {
            mu,
            sigma_squared,
            rank,
            team: team.clone(),
        });
    }

    let mut score_map = HashMap::new();
    if let Some(score_vec) = scores {
        if score_vec.len() == team_infos.len() {
            for (i, score) in score_vec.iter().enumerate() {
                score_map.insert(i, *score);
            }
        }
    }

    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, team) in team_infos.iter().enumerate() {
        rank_groups.entry(team.rank).or_default().push(i);
    }

    for (i, team_i) in team_infos.iter().enumerate() {
        let mut omega = 0.0;
        let mut delta = 0.0;

        for (q, team_q) in team_infos.iter().enumerate() {
            if q == i {
                continue;
            }

            let mut margin_factor = 1.0;
            if let Some(score_i) = score_map.get(&i) {
                if let Some(score_q) = score_map.get(&q) {
                    let score_diff = (score_i - score_q).abs();
                    if score_diff > 0.0 && team_q.rank < team_i.rank && margin > 0.0 {
                        margin_factor = (1.0 + score_diff / margin).ln();
                    }
                }
            }

            let c_iq = (team_i.sigma_squared + team_q.sigma_squared + 2.0 * beta.powi(2)).sqrt();
            let piq = 1.0 / (1.0 + ((team_q.mu - team_i.mu) * margin_factor / c_iq).exp());
            let sigma_sq_c = team_i.sigma_squared / c_iq;

            let s = if team_q.rank > team_i.rank {
                1.0
            } else if team_q.rank == team_i.rank {
                0.5
            } else {
                0.0
            };

            omega += sigma_sq_c * (s - piq);

            let gamma = team_i.sigma_squared.sqrt() / c_iq;
            delta += (gamma * sigma_sq_c / c_iq) * piq * (1.0 - piq);
        }

        let mut new_team: Vec<SkillRating> = Vec::with_capacity(team_i.team.len());
        for (j, player) in team_i.team.iter().enumerate() {
            let weight = weights.map_or(1.0, |w| w[i][j]);
            let sigma_sq = player.sigma.powi(2);

            let mu = if omega >= 0.0 {
                player.mu + (sigma_sq / team_i.sigma_squared) * omega * weight
            } else {
                player.mu + (sigma_sq / team_i.sigma_squared) * omega / weight
            };

            let sigma_scale = if omega >= 0.0 {
                1.0 - (sigma_sq / team_i.sigma_squared) * delta * weight
            } else {
                1.0 - (sigma_sq / team_i.sigma_squared) * delta / weight
            };

            let sigma = player.sigma * sigma_scale.max(kappa).sqrt();
            new_team.push(SkillRating::new(mu, sigma));
        }

        result.push(new_team);
    }

    // Adjust for tied ranks
    for (_, indices) in rank_groups {
        if indices.len() > 1 {
            let avg_mu_change: f64 = indices
                .iter()
                .map(|&i| result[i][0].mu - teams[i][0].mu)
                .sum::<f64>()
                / indices.len() as f64;

            for &i in &indices {
                for j in 0..result[i].len() {
                    result[i][j].mu = teams[i][j].mu + avg_mu_change;
                }
            }
        }
    }

    result
}

fn _convert_scores_to_ranks(
    teams: &[Vec<SkillRating>],
    scores: &Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    let scores = scores.as_ref().unwrap();
    let ranks = scores.iter().map(|s| -s).collect();
    let ranks = _calculate_rankings(teams, ranks);
    Some(ranks)
}

/// Default gamma function for Bradley-Terry Full Pairing.
///
/// `c`: The square root of the collective team sigma.
/// `_k`: The number of teams in the game.
/// `_mu`: The mean of the team's rating.
/// `sigma_squared``: The variance of the team's rating.
/// `team`: The team rating object.
/// `rank`: The rank of the team.
/// `weights`: The weights of the players in a team.
///
/// Return: A number
fn _gamma(
    c: f64,
    _k: usize,
    _mu: f64,
    sigma_squared: f64,
    _team: &[SkillRating],
    _rank: usize,
    _weights: Option<&[f64]>,
) -> f64 {
    sigma_squared.sqrt() / c
}

fn _normalize(vector: &[f64], target_minimum: f64, target_maximum: f64) -> Vec<f64> {
    if vector.len() == 1 {
        return vec![target_maximum];
    }

    let source_minimum = vector.iter().cloned().fold(f64::INFINITY, f64::min);
    let source_maximum = vector.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut source_range = source_maximum - source_minimum;
    let target_range = target_maximum - target_minimum;

    if source_range == 0.0 {
        source_range = 0.0001;
    }

    vector
        .iter()
        .map(|&value| (((value - source_minimum) / source_range) * target_range) + target_minimum)
        .collect()
}

/// Reorders `objects` according to the values in `tenet`, and returns:
/// - The sorted `objects`
/// - The reordered `tenet` values used for restoring the original order
fn _unwind<T: Clone>(tenet: &[f64], objects: &[T]) -> (Vec<T>, Vec<f64>) {
    // Build a Vec of (tenet_value, object, index)
    let mut zipped: Vec<(f64, T, usize)> = tenet
        .iter()
        .cloned()
        .zip(objects.iter().cloned())
        .enumerate()
        .map(|(i, (t, o))| (t, o, i))
        .collect();

    // Sort by tenet ascending, break ties with original index for stability
    zipped.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });

    // Extract the sorted objects and their reordered tenet values
    let sorted_objects = zipped.iter().map(|(_, obj, _)| obj.clone()).collect();
    let reordered_tenets = zipped.iter().map(|(t, _, _)| *t).collect();

    (sorted_objects, reordered_tenets)
}

// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_succeed() {
        assert!(true);
    }

    #[test]
    fn test_new() {
        let algo = SkillAlgorithm::new(
            25.0,
            25.0 / 3.0,
            25.0 / 6.0,
            0.0001,
            25.0 / 300.0,
            0.0,
            false,
            false,
        );
        assert!(approx_equal(algo.mu, 25.0, 1e-5));
        assert!(approx_equal(algo.sigma, 25.0 / 3.0, 1e-5));
        assert!(approx_equal(algo.beta, 25.0 / 6.0, 1e-5));
        assert!(approx_equal(algo.kappa, 0.0001, 1e-5));
        assert!(approx_equal(algo.tau, 25.0 / 300.0, 1e-5));
        assert!(approx_equal(algo.margin, 0.0, 1e-5));
        assert!(!algo.limit_sigma);
        assert!(!algo.balance);
    }

    #[test]
    fn test_rate_not_enough_teams() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players);
        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, None)
            .expect_err("how did we rate a single team");
    }

    #[test]
    fn test_rate_not_enough_players() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players);
        teams.push(Vec::new());
        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, None)
            .expect_err("team 1 had zero players");
    }

    #[test]
    fn test_rate_ranks_and_scores_supplied() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let ranks = Vec::new();
        let scores = Vec::new();

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, Some(ranks), Some(scores), None)
            .expect_err("ranks and scores supplied");
    }

    #[test]
    fn test_rate_too_few_ranks() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let ranks = Vec::new();

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, Some(ranks), None, None)
            .expect_err("not enough ranks");
    }

    #[test]
    fn test_rate_too_many_ranks() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let mut ranks = Vec::new();
        ranks.push(1.0);
        ranks.push(2.0);
        ranks.push(3.0);

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, Some(ranks), None, None)
            .expect_err("too many ranks");
    }

    #[test]
    fn test_rate_too_few_scores() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let scores = Vec::new();

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, Some(scores), None)
            .expect_err("not enough scores");
    }

    #[test]
    fn test_rate_too_many_scores() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let mut scores = Vec::new();
        scores.push(1.0);
        scores.push(2.0);
        scores.push(3.0);

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, Some(scores), None)
            .expect_err("too many scores");
    }

    #[test]
    fn test_rate_too_few_team_weights() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let team_weights = Vec::new();

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, Some(team_weights))
            .expect_err("not enough team weights");
    }

    #[test]
    fn test_rate_too_many_team_weights() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let mut team_weights = Vec::new();
        let player_weights = Vec::new();
        team_weights.push(player_weights.clone());
        team_weights.push(player_weights.clone());
        team_weights.push(player_weights);

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, Some(team_weights))
            .expect_err("too many team weights");
    }

    #[test]
    fn test_rate_too_few_player_weights() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let mut team_weights = Vec::new();
        let mut player_weights = Vec::new();
        player_weights.push(50.0);
        team_weights.push(player_weights);
        team_weights.push(Vec::new());

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, Some(team_weights))
            .expect_err("not enough player weights");
    }

    #[test]
    fn test_rate_too_many_player_weights() {
        let mut players = Vec::new();
        players.push(SkillRating::default());
        let mut teams = Vec::new();
        teams.push(players.clone());
        teams.push(players);

        let mut team_weights = Vec::new();
        let mut player_weights = Vec::new();
        player_weights.push(50.0);
        team_weights.push(player_weights.clone());
        player_weights.push(50.0);
        team_weights.push(player_weights);

        let algo = SkillAlgorithm::default();
        algo.rate(&teams, None, None, Some(team_weights))
            .expect_err("too many player weights");
    }

    #[test]
    fn test_rate_1v1_000s() {
        // import openskill.models
        // algo = openskill.models.BradleyTerryFull()
        // player1 = algo.rating()
        // player2 = algo.rating()
        // algo.rate([[player1], [player2]])

        let teams = vec![vec![SkillRating::default()], vec![SkillRating::default()]];
        let algo = SkillAlgorithm::default();
        let result = algo.rate(&teams, None, None, None).expect("1v1 match 000");

        assert_eq!(2, result.len());
        let player1 = &result[0][0];
        let player2 = &result[1][0];
        assert!(approx_equal(27.635389493140497, player1.mu, 1e-5));
        assert!(approx_equal(8.06590141354368, player1.sigma, 1e-5));
        assert!(approx_equal(22.364610506859503, player2.mu, 1e-5));
        assert!(approx_equal(8.06590141354368, player2.sigma, 1e-5));
    }

    fn approx_equal(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }
}
