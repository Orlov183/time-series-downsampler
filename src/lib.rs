//! Downsample high-frequency numeric observations into fixed-interval buckets.
//!
//! The core idea: you have a stream of `(timestamp, value)` samples, and you want
//! one row per fixed time window containing min, max, sum and count. This is the
//! shape most charting backends want, and it is cheap to compute in one pass.
//!
//! Design decision: buckets are keyed by `timestamp / interval` using integer
//! division, so a bucket's identity is *the start of the window it covers*.
//! Timestamps are `i64` milliseconds; that is the smallest unit that avoids
//! floating-point bucket boundaries and is large enough for any realistic range.
//! Negative timestamps are allowed and floor toward negative infinity via
//! `div_euclid`, so samples before the Unix epoch land in the bucket you would
//! draw on a chart.
//!
//! Aggregates are accumulated in `f64`. Summing many `f64` values is not
//! associative, but the alternative (Kahan summation or integer fixed-point)
//! would either complicate the API or silently overflow. We document the
//! floating-point behaviour instead of pretending it does not exist.

/// One aggregated fixed-interval bucket.
///
/// `start` is inclusive and `end` is exclusive, both in the same unit as the
/// timestamps fed in. `count` is the number of samples folded into the bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct Bucket {
    /// Inclusive start of the interval, in timestamp units.
    pub start: i64,
    /// Exclusive end of the interval, in timestamp units.
    pub end: i64,
    /// Smallest value observed in the interval.
    pub min: f64,
    /// Largest value observed in the interval.
    pub max: f64,
    /// Sum of all values in the interval.
    pub sum: f64,
    /// Number of samples in the interval. Always at least 1 for a returned bucket.
    pub count: u64,
}

/// Streaming downsampler that folds `(timestamp, value)` samples into buckets.
///
/// Buckets are emitted in timestamp order. Samples may be supplied out of order;
/// the downsampler keeps a map of open buckets and sorts the result on `finish`.
/// Samples sharing a timestamp are simply added to the same bucket.
#[derive(Debug, Clone)]
pub struct Downsampler {
    interval: i64,
    // Keyed by bucket start. Using a Vec of (start, accumulator) would be fine
    // for small inputs, but a BTreeMap gives ordered iteration for free and
    // avoids a separate sort pass for the common in-order case.
    buckets: std::collections::BTreeMap<i64, Accum>,
}

#[derive(Debug, Clone)]
struct Accum {
    min: f64,
    max: f64,
    sum: f64,
    count: u64,
}

impl Accum {
    fn new(value: f64) -> Self {
        Accum {
            min: value,
            max: value,
            sum: value,
            count: 1,
        }
    }

    fn push(&mut self, value: f64) {
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.sum += value;
        self.count += 1;
    }
}

impl Downsampler {
    /// Create a downsampler with a fixed bucket width.
    ///
    /// # Panics
    ///
    /// Panics if `interval <= 0`. A zero or negative interval has no sensible
    /// bucket meaning, and silently substituting a default would hide a caller
    /// bug.
    pub fn new(interval: i64) -> Self {
        assert!(interval > 0, "interval must be positive, got {interval}");
        Downsampler {
            interval,
            buckets: std::collections::BTreeMap::new(),
        }
    }

    /// The bucket width this downsampler was constructed with.
    pub fn interval(&self) -> i64 {
        self.interval
    }

    /// Add a single observation.
    ///
    /// `NaN` values are rejected by returning `false`: a NaN would poison both
    /// `min` and `max` comparisons (every comparison against NaN is false), so
    /// the bucket would silently keep whatever it had before. Refusing them at
    /// the boundary keeps the invariant that `min <= max` always holds.
    pub fn add(&mut self, timestamp: i64, value: f64) -> bool {
        if value.is_nan() {
            return false;
        }
        let start = bucket_start(timestamp, self.interval);
        match self.buckets.get_mut(&start) {
            Some(acc) => acc.push(value),
            None => {
                self.buckets.insert(start, Accum::new(value));
            }
        }
        true
    }

    /// Add every sample from an iterator of `(timestamp, value)` pairs.
    ///
    /// Returns the number of samples that were actually accepted (i.e. not
    /// rejected as NaN).
    pub fn extend<I>(&mut self, samples: I) -> usize
    where
        I: IntoIterator<Item = (i64, f64)>,
    {
        let mut accepted = 0;
        for (ts, v) in samples {
            if self.add(ts, v) {
                accepted += 1;
            }
        }
        accepted
    }

    /// Number of buckets currently held.
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// True when no samples have been accepted.
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Consume the downsampler and return the buckets in ascending start order.
    pub fn finish(self) -> Vec<Bucket> {
        self.buckets
            .into_iter()
            .map(|(start, acc)| Bucket {
                start,
                end: start + self.interval,
                min: acc.min,
                max: acc.max,
                sum: acc.sum,
                count: acc.count,
            })
            .collect()
    }
}

/// Compute the inclusive start of the bucket containing `timestamp`.
///
/// Uses `div_euclid` rather than `/` so that negative timestamps floor toward
/// negative infinity. With truncating division, `-1 / 1000 == 0`, which would
/// place a sample at `-1` into the bucket starting at `0` and produce a bucket
/// whose `start > timestamp`. That is surprising on a chart.
fn bucket_start(timestamp: i64, interval: i64) -> i64 {
    timestamp.div_euclid(interval) * interval
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn empty_downsampler_finishes_empty() {
        let d = Downsampler::new(10);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
        assert_eq!(d.finish(), Vec::<Bucket>::new());
    }

    #[test]
    fn interval_is_exposed() {
        let d = Downsampler::new(250);
        assert_eq!(d.interval(), 250);
    }

    #[test]
    #[should_panic(expected = "interval must be positive")]
    fn zero_interval_panics() {
        let _ = Downsampler::new(0);
    }

    #[test]
    #[should_panic(expected = "interval must be positive")]
    fn negative_interval_panics() {
        let _ = Downsampler::new(-5);
    }

    #[test]
    fn single_sample_forms_one_bucket() {
        let mut d = Downsampler::new(100);
        assert!(d.add(42, 3.5));
        let out = d.finish();
        assert_eq!(out.len(), 1);
        let b = &out[0];
        assert_eq!(b.start, 0);
        assert_eq!(b.end, 100);
        assert!(approx(b.min, 3.5));
        assert!(approx(b.max, 3.5));
        assert!(approx(b.sum, 3.5));
        assert_eq!(b.count, 1);
    }

    #[test]
    fn aggregates_min_max_sum_count() {
        let mut d = Downsampler::new(1000);
        for (ts, v) in [(0, 5.0), (10, -2.0), (999, 7.0), (1000, 1.0)] {
            assert!(d.add(ts, v));
        }
        let out = d.finish();
        assert_eq!(out.len(), 2);

        let first = &out[0];
        assert_eq!(first.start, 0);
        assert_eq!(first.end, 1000);
        assert!(approx(first.min, -2.0));
        assert!(approx(first.max, 7.0));
        assert!(approx(first.sum, 10.0));
        assert_eq!(first.count, 3);

        let second = &out[1];
        assert_eq!(second.start, 1000);
        assert_eq!(second.end, 2000);
        assert!(approx(second.min, 1.0));
        assert!(approx(second.max, 1.0));
        assert!(approx(second.sum, 1.0));
        assert_eq!(second.count, 1);
    }

    #[test]
    fn out_of_order_samples_land_in_correct_buckets() {
        let mut d = Downsampler::new(10);
        assert!(d.add(25, 2.0));
        assert!(d.add(3, 1.0));
        assert!(d.add(15, 3.0));
        let out = d.finish();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[1].start, 10);
        assert_eq!(out[2].start, 20);
        assert!(approx(out[0].sum, 1.0));
        assert!(approx(out[1].sum, 3.0));
        assert!(approx(out[2].sum, 2.0));
    }

    #[test]
    fn duplicate_timestamps_accumulate() {
        let mut d = Downsampler::new(100);
        assert!(d.add(50, 1.0));
        assert!(d.add(50, 2.0));
        assert!(d.add(50, 3.0));
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].count, 3);
        assert!(approx(out[0].sum, 6.0));
        assert!(approx(out[0].min, 1.0));
        assert!(approx(out[0].max, 3.0));
    }

    #[test]
    fn negative_timestamps_floor_to_negative_bucket_starts() {
        let mut d = Downsampler::new(100);
        // -1 belongs in [-100, 0), not [0, 100).
        assert!(d.add(-1, 4.0));
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, -100);
        assert_eq!(out[0].end, 0);
        assert!(out[0].start <= -1 && -1 < out[0].end);
    }

    #[test]
    fn exact_boundary_opens_new_bucket() {
        let mut d = Downsampler::new(100);
        assert!(d.add(99, 1.0));
        assert!(d.add(100, 2.0));
        let out = d.finish();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[1].start, 100);
    }

    #[test]
    fn nan_is_rejected_and_does_not_create_bucket() {
        let mut d = Downsampler::new(10);
        assert!(!d.add(5, f64::NAN));
        assert!(d.is_empty());
        assert_eq!(d.extend([(5, f64::NAN), (6, 1.0)]), 1);
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].count, 1);
        assert!(approx(out[0].min, 1.0));
    }

    #[test]
    fn infinities_are_accepted() {
        let mut d = Downsampler::new(10);
        assert!(d.add(1, f64::INFINITY));
        assert!(d.add(2, f64::NEG_INFINITY));
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert!(out[0].min.is_infinite() && out[0].min < 0.0);
        assert!(out[0].max.is_infinite() && out[0].max > 0.0);
        assert!(out[0].sum.is_nan(), "inf + -inf is NaN, and we surface that");
    }

    #[test]
    fn extend_returns_accepted_count() {
        let mut d = Downsampler::new(100);
        let accepted = d.extend([(0, 1.0), (1, 2.0), (2, f64::NAN), (3, 3.0)]);
        assert_eq!(accepted, 3);
        let out = d.finish();
        assert_eq!(out[0].count, 3);
    }

    #[test]
    fn buckets_are_sorted_ascending() {
        let mut d = Downsampler::new(10);
        for ts in [50, 10, 30, 20, 40] {
            assert!(d.add(ts, 1.0));
        }
        let out = d.finish();
        let starts: Vec<i64> = out.iter().map(|b| b.start).collect();
        assert_eq!(starts, vec![10, 20, 30, 40, 50]);
        for pair in out.windows(2) {
            assert!(pair[0].start < pair[1].start);
            assert_eq!(pair[0].end, pair[1].start);
        }
    }

    #[test]
    fn len_and_is_empty_track_buckets_not_samples() {
        let mut d = Downsampler::new(100);
        assert!(d.is_empty());
        assert!(d.add(1, 1.0));
        assert!(d.add(2, 2.0));
        assert_eq!(d.len(), 1);
        assert!(!d.is_empty());
        assert!(d.add(200, 3.0));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn large_interval_collapses_everything() {
        let mut d = Downsampler::new(i64::MAX / 4);
        assert!(d.add(0, 1.0));
        assert!(d.add(1_000_000, 2.0));
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].count, 2);
        assert!(approx(out[0].sum, 3.0));
    }
}
