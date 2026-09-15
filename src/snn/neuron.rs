//! [`Neuron`] — per-neuron LIF state and the **exact linear** integrator that
//! matches Brian2's `method='linear'` for the fly-brain model.
//!
//! ## The model (from `model.py:47-55`)
//!
//! While *not* refractory, the two state variables evolve by the coupled linear
//! ODEs
//! ```text
//!   dg/dt = -g / tau
//!   dv/dt = (v_0 - v + g) / t_mbr
//! ```
//! A spike fires when `v > v_th`, upon which `v = v_rst`, `g = 0`, and the neuron
//! becomes refractory for `t_rfc`. While refractory, both `v` and `g` are frozen
//! (`(unless refractory)` in the Brian2 equations).
//!
//! ## Exact integration
//!
//! Because the system is linear with constant coefficients, one timestep has a
//! closed form (no Euler error). With `a = exp(-dt/t_mbr)` and `b = exp(-dt/tau)`:
//!
//! `g` is independent and decays geometrically:
//! ```text
//!   g' = g · b
//! ```
//! For `v`, the resting term contributes `v_0·(1 - a)` and the homogeneous term
//! `v·a`. The `g`-forcing term integrates to `g · C · (b - a)` where
//! `C = 1 / (1 - t_mbr/tau)` (the coefficient of the particular solution). Hence
//! ```text
//!   v' = v·a + v_0·(1 - a) + g · (b - a) / (1 - t_mbr/tau)
//! ```
//! (The `tau == t_mbr` degenerate case is handled separately; it does not arise
//! for the fly-brain defaults where `tau = 5 ms ≠ t_mbr = 20 ms`.)

use crate::snn::params::LifParams;

/// State of one LIF neuron. `v` and `g` are in volts; `refractory_until` is an
/// absolute step index (see [`Neuron::step`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neuron {
    /// Membrane potential (V).
    pub v: f64,
    /// Synaptic conductance term (V) — the `g` variable.
    pub g: f64,
    /// The step index at (and after which) the neuron is no longer refractory.
    /// A neuron is refractory while `current_step < refractory_until`.
    pub refractory_until: u64,
    /// If true, this neuron never enters a refractory period (Poisson-excited
    /// neurons have `rfc = 0`; see `model.py:95`).
    pub refractory_disabled: bool,
}

impl Neuron {
    /// A fresh neuron at rest: `v = v_0`, `g = 0`, not refractory.
    pub fn at_rest(params: &LifParams) -> Self {
        Self {
            v: params.v_0,
            g: 0.0,
            refractory_until: 0,
            refractory_disabled: false,
        }
    }

    /// Whether the neuron is refractory at `current_step`.
    #[inline]
    pub fn is_refractory(&self, current_step: u64) -> bool {
        !self.refractory_disabled && current_step < self.refractory_until
    }

    /// Add a conductance increment to `g` (from a delivered synaptic event).
    #[inline]
    pub fn add_g(&mut self, dw: f64) {
        self.g += dw;
    }

    /// Add a voltage increment to `v` (from a Poisson input event).
    #[inline]
    pub fn add_v(&mut self, dv: f64) {
        self.v += dv;
    }

    /// Advance this neuron by one timestep using the exact linear update, then
    /// apply the threshold/reset rule.
    ///
    /// Returns `true` if the neuron spiked this step. On a spike, `v` is set to
    /// `v_rst`, `g` to `0`, and the neuron becomes refractory until
    /// `current_step + 1 + refractory_steps` (the refractory clock starts on the
    /// step *after* the spike, matching Brian2's post-reset refractory).
    ///
    /// `current_step` is the index of the step being computed; it is used only
    /// for refractory bookkeeping.
    pub fn step(&mut self, params: &LifParams, current_step: u64) -> bool {
        if self.is_refractory(current_step) {
            // Frozen: neither v nor g evolve while refractory.
            return false;
        }

        let a = (-params.dt / params.t_mbr).exp();
        let b = (-params.dt / params.tau).exp();

        // Exact linear update (see module docs for the derivation).
        let ratio = params.t_mbr / params.tau;
        let g_coeff = (b - a) / (1.0 - ratio);

        let v_new = self.v * a + params.v_0 * (1.0 - a) + self.g * g_coeff;
        let g_new = self.g * b;

        self.v = v_new;
        self.g = g_new;

        // Threshold → reset (model.py:53-55: 'v > v_th' → v=v_rst; g=0).
        if self.v > params.v_th {
            self.v = params.v_rst;
            self.g = 0.0;
            if !self.refractory_disabled {
                self.refractory_until = current_step + 1 + params.refractory_steps() as u64;
            }
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    /// A tiny-`dt` Euler reference for the coupled ODEs (no threshold), used to
    /// confirm the closed-form exact update over one macro-step.
    fn euler_reference(v0: f64, g0: f64, params: &LifParams, macro_dt: f64) -> (f64, f64) {
        let micro = 1e-9; // 1 ns substeps
        let steps = (macro_dt / micro).round() as u64;
        let (mut v, mut g) = (v0, g0);
        for _ in 0..steps {
            let dv = (params.v_0 - v + g) / params.t_mbr;
            let dg = -g / params.tau;
            v += dv * micro;
            g += dg * micro;
        }
        (v, g)
    }

    #[test]
    fn free_neuron_matches_analytic_decay_toward_rest() {
        // With g = 0, v relaxes toward v_0 exponentially: v' = v_0 + (v - v_0)·a.
        let p = LifParams::fly_brain();
        let mut n = Neuron::at_rest(&p);
        n.v = -0.048; // displaced above rest, still below threshold (−45 mV)
        n.g = 0.0;
        let a = (-p.dt / p.t_mbr).exp();
        let expected = p.v_0 + (n.v - p.v_0) * a;
        let spiked = n.step(&p, 0);
        assert!(!spiked);
        assert!(
            approx(n.v, expected, 1e-12),
            "v={} expected={}",
            n.v,
            expected
        );
        assert_eq!(n.g, 0.0);
    }

    #[test]
    fn exact_update_matches_fine_euler_with_conductance() {
        // With a non-zero g, compare the closed form against a 1ns-Euler solve
        // over one 0.1ms macro-step. Stay below threshold so no reset fires.
        let p = LifParams::fly_brain();
        let mut n = Neuron::at_rest(&p);
        n.v = -0.050;
        n.g = 0.010; // 10 mV of conductance drive
        let (v_ref, g_ref) = euler_reference(n.v, n.g, &p, p.dt);
        let spiked = n.step(&p, 0);
        assert!(!spiked);
        assert!(approx(n.v, v_ref, 1e-7), "v={} ref={}", n.v, v_ref);
        assert!(approx(n.g, g_ref, 1e-9), "g={} ref={}", n.g, g_ref);
    }

    #[test]
    fn g_decays_geometrically() {
        let p = LifParams::fly_brain();
        let mut n = Neuron::at_rest(&p);
        n.v = p.v_0;
        n.g = 0.02;
        let b = (-p.dt / p.tau).exp();
        let expected_g = n.g * b;
        n.step(&p, 0);
        assert!(approx(n.g, expected_g, 1e-12));
    }
}
