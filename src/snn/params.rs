//! [`LifParams`] — the LIF model constants, ported verbatim from `fly-brain`'s
//! `code/paper-phil-drosophila/model.py` `default_params` (lines 18–56).
//!
//! ## Source mapping
//!
//! | Rust field    | `model.py` key | Value (source) | SI value here |
//! |---------------|----------------|----------------|---------------|
//! | `v_0`         | `v_0`          | −52 mV         | −0.052 V      |
//! | `v_rst`       | `v_rst`        | −52 mV         | −0.052 V      |
//! | `v_th`        | `v_th`         | −45 mV         | −0.045 V      |
//! | `t_mbr`       | `t_mbr`        | 20 ms          | 0.020 s       |
//! | `tau`         | `tau`          | 5 ms           | 0.005 s       |
//! | `t_rfc`       | `t_rfc`        | 2.2 ms         | 0.0022 s      |
//! | `t_dly`       | `t_dly`        | 1.8 ms         | 0.0018 s      |
//! | `w_syn`       | `w_syn`        | 0.275 mV       | 0.000275 V    |
//! | `f_poi`       | `f_poi`        | 250            | 250.0         |
//! | `dt`          | (Brian2 dt)    | 0.1 ms         | 0.0001 s      |
//!
//! Units: SI throughout (volts, seconds). `f64` is used for fidelity; a later
//! real-time milestone may revisit precision for speed.

/// All constants of the fly-brain LIF model, in SI units (volts, seconds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifParams {
    /// Resting potential `v_0` (V). Source: −52 mV.
    pub v_0: f64,
    /// Reset potential after a spike `v_rst` (V). Source: −52 mV.
    pub v_rst: f64,
    /// Spike threshold `v_th` (V). Source: −45 mV.
    pub v_th: f64,
    /// Membrane time constant `t_mbr` (s). Source: 20 ms.
    pub t_mbr: f64,
    /// Synaptic (conductance) time constant `tau` (s). Source: 5 ms.
    pub tau: f64,
    /// Refractory period `t_rfc` (s). Source: 2.2 ms.
    pub t_rfc: f64,
    /// Synaptic transmission delay `t_dly` (s). Source: 1.8 ms.
    pub t_dly: f64,
    /// Base synaptic weight `w_syn` (V). Source: 0.275 mV.
    pub w_syn: f64,
    /// Poisson synapse scaling factor `f_poi` (dimensionless). Source: 250.
    pub f_poi: f64,
    /// Integration timestep `dt` (s). Brian2 default 0.1 ms.
    pub dt: f64,
}

impl LifParams {
    /// The default fly-brain parameters (`model.py` `default_params`).
    pub const fn fly_brain() -> Self {
        Self {
            v_0: -0.052,
            v_rst: -0.052,
            v_th: -0.045,
            t_mbr: 0.020,
            tau: 0.005,
            t_rfc: 0.0022,
            t_dly: 0.0018,
            w_syn: 0.000275,
            f_poi: 250.0,
            dt: 0.0001,
        }
    }

    /// The voltage a single Poisson event adds directly to a target neuron's
    /// membrane `v`: `w_syn · f_poi` (source: `PoissonInput(..., weight=w_syn*f_poi)`,
    /// `model.py:88-95`). With defaults this is 0.275 mV × 250 = 68.75 mV.
    #[inline]
    pub fn poisson_weight(&self) -> f64 {
        self.w_syn * self.f_poi
    }

    /// Delay expressed in whole timesteps: `round(t_dly / dt)`. With defaults
    /// this is round(1.8 ms / 0.1 ms) = 18 steps.
    #[inline]
    pub fn delay_steps(&self) -> usize {
        (self.t_dly / self.dt).round() as usize
    }

    /// Refractory period expressed in whole timesteps: `round(t_rfc / dt)`.
    /// With defaults this is round(2.2 ms / 0.1 ms) = 22 steps.
    #[inline]
    pub fn refractory_steps(&self) -> usize {
        (self.t_rfc / self.dt).round() as usize
    }
}

impl Default for LifParams {
    fn default() -> Self {
        Self::fly_brain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_source_values() {
        let p = LifParams::fly_brain();
        assert_eq!(p.v_0, -0.052);
        assert_eq!(p.v_th, -0.045);
        assert_eq!(p.t_mbr, 0.020);
        assert_eq!(p.tau, 0.005);
        assert_eq!(p.dt, 0.0001);
    }

    #[test]
    fn poisson_weight_is_68_75_mv() {
        let p = LifParams::fly_brain();
        assert!((p.poisson_weight() - 0.06875).abs() < 1e-12);
    }

    #[test]
    fn delay_and_refractory_step_counts() {
        let p = LifParams::fly_brain();
        assert_eq!(p.delay_steps(), 18);
        assert_eq!(p.refractory_steps(), 22);
    }
}
