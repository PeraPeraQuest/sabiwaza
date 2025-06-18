// rating.rs
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

use serde::{Deserialize, Serialize};

/// Represents a player's skill rating
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SkillRating {
    /// Represents the initial belief about the skill of
    /// a player before any matches have been played. Known
    /// mostly as the mean of the Guassian prior distribution.
    pub mu: f64,

    /// Standard deviation of the prior distribution of player.
    pub sigma: f64,
}

impl Default for SkillRating {
    fn default() -> Self {
        Self {
            mu: 25.0,
            sigma: 25.0 / 3.0,
        }
    }
}

impl SkillRating {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    /// Computes a single scalar value representing the player's skill,
    /// where their true skill is `z` standard deviations below their mean.
    ///
    /// This is commonly used to rank players with a specified confidence level,
    /// such as the 99.7% interval corresponding to `z = 3.0`.
    ///
    /// # Arguments
    ///
    /// - `z`: Number of standard deviations to subtract from the mean.
    ///   Typically `3.0` for a 99.7% confidence interval.
    /// - `alpha`: Scaling factor applied to the result. Default is `1.0`.
    /// - `target`: A value to shift the ordinal score toward. Used with `alpha`.
    ///
    /// # Returns
    ///
    /// A single `f64` scalar:  
    /// `α · ((μ - z · σ) + target / α)`
    pub fn ordinal(&self, z: f64, alpha: f64, target: f64) -> f64 {
        alpha * ((self.mu - z * self.sigma) + (target / alpha))
    }
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
    fn test_ordinal_known_values() {
        let z = 3.0;
        let alpha = 1.0;
        let target = 0.0;

        let rating = SkillRating::new(25.0, 25.0 / 3.0);
        let result = rating.ordinal(z, alpha, target);
        let expected = alpha * ((25.0 - z * (25.0 / 3.0)) + (target / alpha));
        assert!(
            (result - expected).abs() < 1e-6,
            "ordinal calculation mismatch"
        );
    }

    #[test]
    fn test_ordinal_monotonicity() {
        let z = 3.0;
        let alpha = 1.0;
        let target = 0.0;

        for i in 0..100 {
            let mu = 10.0 + i as f64 * 0.1;
            let sigma = 2.0 + (i % 10) as f64 * 0.1;

            let rating1 = SkillRating::new(mu, sigma);
            let rating2 = SkillRating::new(mu + 0.01, sigma);
            let rating3 = SkillRating::new(mu, sigma + 0.01);

            let ord1 = rating1.ordinal(z, alpha, target);
            let ord2 = rating2.ordinal(z, alpha, target);
            let ord3 = rating3.ordinal(z, alpha, target);

            assert!(ord2 > ord1, "ordinal should increase with mu");
            assert!(ord3 < ord1, "ordinal should decrease with sigma");
            assert!(ord1.is_finite(), "ordinal should be finite");
        }
    }
}
