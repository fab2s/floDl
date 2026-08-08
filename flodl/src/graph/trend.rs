/// Statistical analysis over a scalar time series.
///
/// Created by `Graph::trend()` from epoch history, or directly via `Trend::new`.
pub struct Trend {
    values: Vec<f64>,
}

impl Trend {
    /// Create a trend from a vector of scalar observations (one per epoch).
    pub fn new(values: Vec<f64>) -> Self {
        Trend { values }
    }

    /// Number of recorded data points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True if no data points have been recorded.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// All recorded values as a slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Last n values (or all if n > len).
    pub fn last(&self, n: usize) -> &[f64] {
        let start = self.values.len().saturating_sub(n);
        &self.values[start..]
    }

    /// Most recent value, or 0.0 if empty.
    pub fn latest(&self) -> f64 {
        self.values.last().copied().unwrap_or(0.0)
    }

    /// Arithmetic mean of all values, or 0.0 if empty.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Minimum value, or `+inf` if empty.
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Maximum value, or `-inf` if empty.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// OLS linear regression slope. window=0 means all points.
    /// Negative slope = decreasing (improving for losses).
    pub fn slope(&self, window: usize) -> f64 {
        let data = if window > 0 {
            self.last(window)
        } else {
            &self.values
        };
        ols_slope(data)
    }

    /// True if |slope| < tol (loss is no longer changing meaningfully).
    pub fn stalled(&self, window: usize, tol: f64) -> bool {
        self.slope(window).abs() < tol
    }

    /// True if slope < 0 (loss is decreasing).
    pub fn improving(&self, window: usize) -> bool {
        self.slope(window) < 0.0
    }

    /// True if variance < tol over the window (values have stabilized).
    pub fn converged(&self, window: usize, tol: f64) -> bool {
        let data = if window > 0 {
            self.last(window)
        } else {
            &self.values
        };
        if data.len() < 2 {
            return false;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;
        var < tol
    }
}

/// Ordinary least squares slope for evenly-spaced data.
fn ols_slope(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();

    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

/// A collection of Trends for group queries.
pub struct TrendGroup(pub Vec<Trend>);

impl TrendGroup {
    /// Number of trends in the group.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True if the group contains no trends.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// True if every trend in the group has a negative slope (decreasing loss).
    pub fn all_improving(&self, window: usize) -> bool {
        !self.0.is_empty() && self.0.iter().all(|t| t.improving(window))
    }

    /// True if at least one trend has a negative slope.
    pub fn any_improving(&self, window: usize) -> bool {
        self.0.iter().any(|t| t.improving(window))
    }

    /// True if every trend is stalled (|slope| < tol).
    pub fn all_stalled(&self, window: usize, tol: f64) -> bool {
        !self.0.is_empty() && self.0.iter().all(|t| t.stalled(window, tol))
    }

    /// True if at least one trend is stalled (|slope| < tol).
    pub fn any_stalled(&self, window: usize, tol: f64) -> bool {
        self.0.iter().any(|t| t.stalled(window, tol))
    }

    /// True if every trend has converged (variance < tol over window).
    pub fn all_converged(&self, window: usize, tol: f64) -> bool {
        !self.0.is_empty() && self.0.iter().all(|t| t.converged(window, tol))
    }

    /// True if at least one trend has converged (variance < tol over window).
    pub fn any_converged(&self, window: usize, tol: f64) -> bool {
        self.0.iter().any(|t| t.converged(window, tol))
    }

    /// Average OLS slope across all trends in the group.
    pub fn mean_slope(&self, window: usize) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.0.iter().map(|t| t.slope(window)).sum();
        sum / self.0.len() as f64
    }

    /// Per-trend OLS slopes, in the same order as the group.
    pub fn slopes(&self, window: usize) -> Vec<f64> {
        self.0.iter().map(|t| t.slope(window)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_basic() {
        let t = Trend::new(vec![10.0, 8.0, 6.0, 4.0, 2.0]);
        assert_eq!(t.len(), 5);
        assert!(!t.is_empty());
        assert!((t.latest() - 2.0).abs() < 1e-10);
        assert!((t.mean() - 6.0).abs() < 1e-10);
        assert!((t.min() - 2.0).abs() < 1e-10);
        assert!((t.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_trend_last() {
        let t = Trend::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(t.last(3), &[3.0, 4.0, 5.0]);
        assert_eq!(t.last(10), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(t.last(0), &[] as &[f64]);
    }

    #[test]
    fn test_trend_slope_decreasing() {
        // Perfectly linear decrease: slope should be -2
        let t = Trend::new(vec![10.0, 8.0, 6.0, 4.0, 2.0]);
        let slope = t.slope(0);
        assert!((slope - (-2.0)).abs() < 1e-10, "slope={}", slope);
        assert!(t.improving(0));
        assert!(!t.stalled(0, 0.1));
    }

    #[test]
    fn test_trend_slope_increasing() {
        let t = Trend::new(vec![1.0, 3.0, 5.0, 7.0]);
        let slope = t.slope(0);
        assert!((slope - 2.0).abs() < 1e-10, "slope={}", slope);
        assert!(!t.improving(0));
    }

    #[test]
    fn test_trend_slope_flat() {
        let t = Trend::new(vec![5.0, 5.0, 5.0, 5.0]);
        assert!((t.slope(0)).abs() < 1e-10);
        assert!(t.stalled(0, 0.01));
    }

    #[test]
    fn test_trend_slope_window() {
        // First half increasing, last 3 decreasing
        let t = Trend::new(vec![1.0, 5.0, 10.0, 8.0, 6.0]);
        assert!(t.slope(0) > 0.0); // overall increasing
        assert!(t.slope(3) < 0.0); // last 3 decreasing
        assert!(t.improving(3));
    }

    #[test]
    fn test_trend_converged() {
        let t = Trend::new(vec![5.0, 5.01, 4.99, 5.0, 5.01]);
        assert!(t.converged(0, 0.001)); // variance is tiny
        assert!(!t.converged(0, 0.00001)); // too tight
    }

    #[test]
    fn test_trend_empty() {
        let t = Trend::new(vec![]);
        assert!(t.is_empty());
        assert!((t.latest()).abs() < 1e-10);
        assert!((t.mean()).abs() < 1e-10);
        assert!((t.slope(0)).abs() < 1e-10);
        assert!(!t.converged(0, 0.001)); // needs >= 2 points
    }

    #[test]
    fn test_trend_single_point() {
        let t = Trend::new(vec![42.0]);
        assert!((t.latest() - 42.0).abs() < 1e-10);
        assert!((t.slope(0)).abs() < 1e-10);
        assert!(!t.converged(0, 0.001)); // needs >= 2
    }

    #[test]
    fn test_trend_group_all_improving() {
        let g = TrendGroup(vec![
            Trend::new(vec![10.0, 8.0, 6.0]),
            Trend::new(vec![5.0, 3.0, 1.0]),
        ]);
        assert!(g.all_improving(0));
        assert!(g.any_improving(0));
    }

    #[test]
    fn test_trend_group_mixed() {
        let g = TrendGroup(vec![
            Trend::new(vec![10.0, 8.0, 6.0]), // improving
            Trend::new(vec![1.0, 3.0, 5.0]),  // not improving
        ]);
        assert!(!g.all_improving(0));
        assert!(g.any_improving(0));
    }

    #[test]
    fn test_trend_group_slopes() {
        let g = TrendGroup(vec![
            Trend::new(vec![10.0, 8.0, 6.0, 4.0]), // slope = -2
            Trend::new(vec![0.0, 1.0, 2.0, 3.0]),  // slope = 1
        ]);
        let slopes = g.slopes(0);
        assert!((slopes[0] - (-2.0)).abs() < 1e-10);
        assert!((slopes[1] - 1.0).abs() < 1e-10);
        assert!((g.mean_slope(0) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_trend_group_empty() {
        let g = TrendGroup(vec![]);
        assert!(g.is_empty());
        assert!(!g.all_improving(0));
        assert!(!g.any_improving(0));
        assert!((g.mean_slope(0)).abs() < 1e-10);
    }
}
