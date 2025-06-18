// error.rs
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

use thiserror::Error;

// Example convenience alias for result types
pub type Result<T> = std::result::Result<T, SabiwazaError>;

/// Common errors returned by the sabiwaza library.
#[derive(Debug, Error)]
pub enum SabiwazaError {
    /// Raised when an input value is invalid (e.g., negative sigma)
    #[error("invalid value: {0}")]
    ValueError(String),

    /// Raised when a function receives data that cannot be used
    #[error("incompatible data: {0}")]
    DataError(String),

    /// Raised when a requested algorithm or model is unknown
    #[error("unknown model or algorithm: {0}")]
    UnknownModel(String),

    /// Placeholder for future error types
    #[error("unimplemented feature: {0}")]
    Unimplemented(String),
}

// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn test_always_succeed() {
        assert!(true);
    }
}
