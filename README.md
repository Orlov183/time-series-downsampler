# time-series-downsampler

Aggregates high-frequency numeric `(timestamp, value)` observations into
fixed-interval buckets, each carrying `min`, `max`, `sum` and `count`.

```rust
use time_series_downsampler::Downsampler;

let mut d = Downsampler::new(1000); // 1-second buckets, timestamps in ms
assert!(d.add(0, 5.0));
assert!(d.add(250, -2.0));
assert!(d.add(999, 7.0));
assert!(d.add(1000, 1.0));

let buckets = d.finish();
assert_eq!(buckets.len(), 2);
assert_eq!(buckets[0].start, 0);
assert_eq!(buckets[0].end, 1000);
assert_eq!(buckets[0].count, 3);
assert_eq!(buckets[0].min, -2.0);
assert_eq!(buckets[0].max, 7.0);
assert_eq!(buckets[0].sum, 10.0);
assert_eq!(buckets[1].start, 1000);
assert_eq!(buckets[1].count, 1);
```

## Why this exists

Rendering a million raw samples into a 600-pixel chart is wasteful: the browser
throws away almost every point anyway. The usual fix is to pre-aggregate into
fixed windows on the server, where a single pass over the data is cheap. This
library is that single pass, with no allocation per sample beyond the bucket
map.

The trade-off is that buckets are keyed by integer division of the timestamp,
not by a sliding window. That means a sample at `t = 999` and a sample at
`t = 1000` land in different buckets even though they are one millisecond
apart, and there is no way to ask for a bucket that starts at an arbitrary
offset. In exchange, bucket identity is a pure function of the timestamp and
the interval, so results are stable across runs and machines.

## The awkward edge

Timestamps are `i64` and buckets use `div_euclid`, so negative timestamps floor
toward negative infinity. A sample at `t = -1` with a 100-unit interval lands
in the bucket `[-100, 0)`, not `[0, 100)`. Truncating division would have
produced a bucket whose `start` was greater than the timestamp it contained,
which is the kind of thing that only shows up on a chart of pre-epoch data.

`NaN` values are rejected by `add` (which returns `false`) rather than folded
in, because every comparison against `NaN` is false and a `NaN` would silently
leave `min` and `max` at their previous values. Infinities are accepted; note
that `+inf + -inf` sums to `NaN`, and the bucket reports that honestly.

## API

- `Downsampler::new(interval: i64) -> Downsampler` — panics if `interval <= 0`.
- `Downsampler::interval(&self) -> i64`
- `Downsampler::add(&mut self, timestamp: i64, value: f64) -> bool` — `false` if the value was `NaN`.
- `Downsampler::extend<I: IntoIterator<Item = (i64, f64)>>(&mut self, samples: I) -> usize` — returns the number of accepted samples.
- `Downsampler::len(&self) -> usize`, `Downsampler::is_empty(&self) -> bool`
- `Downsampler::finish(self) -> Vec<Bucket>` — buckets in ascending `start` order.
- `Bucket { start: i64, end: i64, min: f64, max: f64, sum: f64, count: u64 }` — `start` inclusive, `end` exclusive.

Samples may be supplied in any order. Duplicate timestamps accumulate into the
same bucket. `sum` is accumulated in `f64`, so very large inputs will drift;
that is documented rather than corrected, because Kahan summation or
fixed-point accumulation would change the public type.
