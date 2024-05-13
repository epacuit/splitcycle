/* User-facing utilities for data processing and interacting with the SplitCycle package */

/// Efficiently store an immutable voting margins graph as a half a square matrix, without its
/// diagonal and without the last row.
/// 
/// Voting margins matrices have reverse diagonal symmetry
/// (i.e. `A[i, j] == -A[j, i]` for all `i, j`) and have zero diagonal entries (since no candidate
/// can defeat or lose to itself). These intrinsic properties allow us to store only the upper half
/// of the full square matrix in memory. This struct provides relevant methods for creating and
/// accessing the data stored in the voting margins graph as if it were a full square matrix.
/// The entry `margins[i, j]` represents the signed margin of victory (positive) or defeat
/// (negative) of candidate `i` against `j`.
pub struct VotingMarginsGraph {
    pub margins: Vec<Vec<isize>>,
    pub n: usize,
}

impl VotingMarginsGraph {
    /// Check if `margins` has the correct dimensions.
    pub fn check_shape(margins: &[Vec<isize>]) -> Result<(), String> {
        // one less than the number of candidates, since last row is omitted
        let n = margins.len();
        for (i, row) in margins.iter().enumerate() {
            if row.len() != (n - i) {
                return Err(format!("incorrect number of candidates in row {}", i));
            }
        }

        Ok(())
    }

    /// Get the equivalent of `margins[i, j]`, treating `margins` as if it were a full square
    /// matrix.
    ///
    /// This skips checking that `i` and `j` are valid keys for `margins`. Use `get` to obtain an
    /// `Option` instead if this is not desired.
    pub fn index(&self, i: usize, j: usize) -> isize {
        if i == j {
            return 0;
        }
        if i < j {
            self.margins[i][j]
        } else {
            -self.margins[j][i]
        }
    }

    /// Safely get the equivalent of `margins[i, j]`, treating `margins` as if it were a full square
    /// matrix.
    ///
    /// This will run checks to ensure that indices `i` and `j` are valid keys for `margins`
    /// (recommended).
    pub fn get(&self, i: usize, j: usize) -> Option<isize> {
        if (i >= self.n) || (j >= self.n) {
            return None;
        }

        Some(self.index(i, j))
    }

    /// Create a new `VotingMarginsGraph` without running recommended checks.
    ///
    /// This is slightly faster than `build` and will never fail, but if the `margins` matrix is
    /// incorrectly formatted, future operations may panic without a correct explanation.
    /// Only use this if the voting margins matrix is guaranteed to be formatted correctly.
    pub fn new(margins: Vec<Vec<isize>>) -> Self {
        let n = margins.len() + 1;
        Self { margins, n }
    }

    /// Recommended way to create a new `VotingMarginsGraph`.
    ///
    /// It will automatically check to ensure that `margins` is correctly formatted.
    pub fn build(margins: Vec<Vec<isize>>) -> Result<Self, String> {
        Self::check_shape(&margins)?;
        Ok(Self::new(margins))
    }
}
