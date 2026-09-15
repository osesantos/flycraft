//! [`PoissonInput`] — deterministic, seeded Poisson stimulation of excited
//! neurons, driving membrane voltage `v` (matching `fly-brain`).
//!
//! ## Semantics (from `model.py:88-95`, `run_brian2_cuda.py:169-174`)
//!
//! Each excited neuron receives a `PoissonInput(target_var='v', N=1, rate=r_poi,
//! weight=w_syn·f_poi)`. So on each timestep an excited neuron fires an input
//! event with probability `p = r_poi · dt` (the Bernoulli approximation of a
//! Poisson process with rate `r_poi` over a small `dt`), and each event adds
//! `w_syn·f_poi` volts directly to `v`. Excited neurons also have their
//! refractory period disabled (`neu[i].rfc = 0`).
//!
//! ## Determinism
//!
//! Randomness comes from a hand-rolled [`SplitMix64`] PRNG seeded once, so a
//! given `(seed, excited set, rate, dt)` yields an identical input spike train
//! across runs — no external RNG crate. Exact spike-by-spike parity with
//! Brian2's own RNG is *not* a goal (M1 validates against the equations and
//! coarse aggregates, not Brian2's random stream).

use crate::snn::params::LifParams;

/// A minimal, fast, deterministic PRNG (SplitMix64). Public so callers can seed
/// reproducible experiments.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator. Any `u64` is a valid seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value (the canonical SplitMix64 step).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next uniform `f64` in `[0, 1)` (53-bit mantissa resolution).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits → [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Deterministic Poisson stimulation of a fixed set of excited neurons.
#[derive(Debug, Clone)]
pub struct PoissonInput {
    /// Indices of neurons receiving Poisson drive.
    excited: Vec<u32>,
    /// Per-step event probability `p = rate · dt`.
    prob: f64,
    /// Voltage added to `v` per event (`w_syn · f_poi`).
    weight: f64,
    rng: SplitMix64,
}

impl PoissonInput {
    /// Create a Poisson input over `excited` neurons at `rate` (Hz), seeded by
    /// `seed`. The per-event voltage and per-step probability are derived from
    /// `params` and `rate`.
    pub fn new(excited: Vec<u32>, rate_hz: f64, params: &LifParams, seed: u64) -> Self {
        Self {
            excited,
            prob: rate_hz * params.dt,
            weight: params.poisson_weight(),
            rng: SplitMix64::new(seed),
        }
    }

    /// The neurons this input excites (which should have refractory disabled).
    pub fn excited(&self) -> &[u32] {
        &self.excited
    }

    /// Advance the input one step: for each excited neuron, with probability
    /// `prob`, add `weight` to that neuron's `v`. `add_v(neuron_index, dv)` is
    /// the caller's closure that applies the increment.
    pub fn step<F: FnMut(usize, f64)>(&mut self, mut add_v: F) {
        // Iterate excited neurons in fixed order so the RNG draw sequence — and
        // therefore the whole input train — is deterministic.
        for &i in &self.excited {
            if self.rng.next_f64() < self.prob {
                add_v(i as usize, self.weight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix64_f64_in_unit_interval() {
        let mut r = SplitMix64::new(7);
        for _ in 0..100_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn poisson_input_reproducible_under_seed() {
        let p = LifParams::fly_brain();
        let make = || PoissonInput::new(vec![0, 1, 2], 150.0, &p, 123);
        let mut a = make();
        let mut b = make();
        let mut events_a = Vec::new();
        let mut events_b = Vec::new();
        for _ in 0..10_000 {
            a.step(|i, dv| events_a.push((i, dv)));
            b.step(|i, dv| events_b.push((i, dv)));
        }
        assert_eq!(events_a, events_b);
        assert!(!events_a.is_empty(), "expected some Poisson events");
    }

    #[test]
    fn poisson_rate_is_in_expected_ballpark() {
        // Over T seconds, a single neuron at rate r should fire ~ r·T events.
        let p = LifParams::fly_brain();
        let rate = 150.0;
        let steps = 200_000; // 200_000 · 0.1ms = 20 s
        let mut input = PoissonInput::new(vec![0], rate, &p, 999);
        let mut count = 0usize;
        for _ in 0..steps {
            input.step(|_, _| count += 1);
        }
        let seconds = steps as f64 * p.dt;
        let measured = count as f64 / seconds;
        // Bernoulli-per-step approximation of a Poisson process; allow 5%.
        assert!(
            (measured - rate).abs() / rate < 0.05,
            "measured rate {measured} Hz vs target {rate} Hz"
        );
    }
}
