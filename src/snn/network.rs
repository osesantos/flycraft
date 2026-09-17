//! [`Network`] — owns neurons, synapses, and (optional) Poisson inputs, and
//! advances the whole fly-brain LIF simulation step by step.
//!
//! ## Per-step ordering (replicating Brian2)
//!
//! For step `s`:
//! 1. **Integrate + threshold + reset**: advance every neuron by the exact
//!    linear update; a neuron with `v > v_th` spikes (→ `v_rst`, `g = 0`,
//!    refractory) and fans out its outgoing synapses for delivery at
//!    `s + delay_steps`.
//! 2. **Deliver** synaptic events scheduled for `s`: add each pending increment
//!    to the target neuron's `g` (these came from spikes `delay_steps` ago).
//! 3. **Poisson input**: add `w_syn·f_poi` to `v` of each excited neuron that
//!    drew an event this step.
//! 4. Clear slot `s`; advance to `s + 1`.
//!
//! Delivery deliberately runs *after* the state update, and is skipped for any
//! neuron that spiked or is currently refractory this step. That matches Brian2,
//! where a delivered event on a spike step is discarded by the reset and `g` is
//! frozen for the whole refractory period. Combined with delivery landing one
//! step after the state update, a spike affects its targets `t_dly` later.
//!
//! This lets the network be **built once and stepped repeatedly** — the property
//! that makes a real-time control loop possible later.

use crate::snn::neuron::Neuron;
use crate::snn::params::LifParams;
use crate::snn::poisson::PoissonInput;
use crate::snn::synapses::Synapses;

/// A recorded spike: `(step, neuron_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeEvent {
    pub step: u64,
    pub neuron: u32,
}

/// The full LIF network.
#[derive(Debug, Clone)]
pub struct Network {
    params: LifParams,
    neurons: Vec<Neuron>,
    synapses: Synapses,
    inputs: Vec<PoissonInput>,
    step_idx: u64,
    /// Recorded spikes (if recording is enabled).
    spikes: Vec<SpikeEvent>,
    recording: bool,
}

impl Network {
    /// Build a network of `n` neurons (all at rest) with the given synapses.
    /// Uses the fly-brain default parameters unless [`Network::with_params`] is
    /// used.
    pub fn new(n: usize, synapses: Synapses) -> Self {
        Self::with_params(LifParams::fly_brain(), n, synapses)
    }

    /// Build a network with explicit parameters.
    pub fn with_params(params: LifParams, n: usize, synapses: Synapses) -> Self {
        assert_eq!(
            n,
            synapses.len(),
            "neuron count must match synapse table size"
        );
        let neurons = vec![Neuron::at_rest(&params); n];
        Self {
            params,
            neurons,
            synapses,
            inputs: Vec::new(),
            step_idx: 0,
            spikes: Vec::new(),
            recording: true,
        }
    }

    /// Attach a Poisson input. Its excited neurons have their refractory period
    /// disabled, matching `neu[i].rfc = 0` in the reference.
    pub fn add_input(&mut self, input: PoissonInput) {
        for &i in input.excited() {
            self.neurons[i as usize].refractory_disabled = true;
        }
        self.inputs.push(input);
    }

    /// Enable or disable spike recording (default: enabled).
    pub fn set_recording(&mut self, on: bool) {
        self.recording = on;
    }

    /// The model parameters.
    pub fn params(&self) -> &LifParams {
        &self.params
    }

    /// Current step index (number of steps advanced so far).
    pub fn current_step(&self) -> u64 {
        self.step_idx
    }

    /// Immutable view of the neurons.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Recorded spikes so far.
    pub fn spikes(&self) -> &[SpikeEvent] {
        &self.spikes
    }

    /// Advance the simulation by exactly one timestep. Returns the number of
    /// neurons that spiked this step.
    pub fn step(&mut self) -> usize {
        let s = self.step_idx;

        // 1. Integrate + threshold + reset; collect spikers.
        let mut spiked_flags = vec![false; self.neurons.len()];
        let mut spikers: Vec<usize> = Vec::new();
        for (i, flag) in spiked_flags.iter_mut().enumerate() {
            if self.neurons[i].step(&self.params, s) {
                *flag = true;
                spikers.push(i);
            }
        }
        for &i in &spikers {
            self.synapses.fan_out_spike(i, s);
            if self.recording {
                self.spikes.push(SpikeEvent {
                    step: s,
                    neuron: i as u32,
                });
            }
        }
        let spiked = spikers.len();

        // 2. Deliver synaptic events scheduled for this step to neurons that
        //    are neither spiking nor refractory. The reference discards events
        //    on a spike step and freezes `g` throughout the refractory period.
        {
            let slot = self.synapses.delivery_slot(s);
            for (i, &dg) in slot.iter().enumerate() {
                if dg != 0.0 && !spiked_flags[i] && !self.neurons[i].is_refractory(s) {
                    self.neurons[i].add_g(dg);
                }
            }
        }
        self.synapses.clear_delivery_slot(s);

        // 3. Poisson input (add to v), likewise discarded on a spiking or
        //    refractory step.
        for input in &mut self.inputs {
            let neurons = &mut self.neurons;
            input.step(|i, dv| {
                if !spiked_flags[i] && !neurons[i].is_refractory(s) {
                    neurons[i].add_v(dv);
                }
            });
        }

        self.step_idx += 1;

        spiked
    }

    /// Advance the simulation for `duration` seconds (rounded to whole steps).
    pub fn run(&mut self, duration_secs: f64) {
        let steps = (duration_secs / self.params.dt).round() as u64;
        for _ in 0..steps {
            self.step();
        }
    }

    /// Mean firing rate (Hz) of neuron `i` over all elapsed simulation time,
    /// computed from recorded spikes. Returns 0 if no time has elapsed.
    pub fn rate_hz(&self, neuron: u32) -> f64 {
        if self.step_idx == 0 {
            return 0.0;
        }
        let count = self.spikes.iter().filter(|e| e.neuron == neuron).count();
        let seconds = self.step_idx as f64 * self.params.dt;
        count as f64 / seconds
    }

    /// Total number of recorded spikes.
    pub fn spike_count(&self) -> usize {
        self.spikes.len()
    }

    /// Number of distinct neurons that spiked at least once.
    pub fn active_neuron_count(&self) -> usize {
        let mut seen = self.spikes.iter().map(|e| e.neuron).collect::<Vec<_>>();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::poisson::PoissonInput;

    /// A network with no synapses (edge-free CSR) of `n` neurons.
    fn isolated(n: usize, delay_steps: usize) -> Network {
        let syn = Synapses::from_edges(n, &[], &[], &[], delay_steps);
        Network::new(n, syn)
    }

    #[test]
    fn isolated_unstimulated_neuron_never_spikes() {
        let mut net = isolated(1, 18);
        net.run(1.0); // 1 s
        assert_eq!(net.spike_count(), 0);
        // Sits at rest.
        assert!((net.neurons()[0].v - net.params().v_0).abs() < 1e-9);
    }

    #[test]
    fn poisson_drive_makes_a_neuron_spike() {
        let mut net = isolated(1, 18);
        // Strong drive: 150 Hz Poisson at 68.75 mV/event will push v over
        // threshold quickly.
        net.add_input(PoissonInput::new(vec![0], 150.0, net.params(), 42));
        net.run(1.0);
        assert!(net.spike_count() > 0, "Poisson-driven neuron should spike");
    }

    #[test]
    fn synaptic_delay_transmits_after_t_dly() {
        // Two neurons: 0 → 1 with a weight large enough to fire neuron 1 in a
        // single g bump. Drive neuron 0 with Poisson; verify neuron 1 only
        // starts spiking at least delay_steps after neuron 0's first spike.
        let p = LifParams::fly_brain();
        // Weight ≥ (v_th - v_0) so one delivered g bump crosses threshold next
        // integrate step: v_th - v_0 = -0.045 - (-0.052) = 7 mV. Use 20 mV.
        let syn = Synapses::from_edges(2, &[0], &[1], &[0.020], p.delay_steps());
        let mut net = Network::with_params(p, 2, syn);
        net.add_input(PoissonInput::new(vec![0], 150.0, net.params(), 7));
        net.run(0.5);

        let first0 = net.spikes().iter().find(|e| e.neuron == 0).map(|e| e.step);
        let first1 = net.spikes().iter().find(|e| e.neuron == 1).map(|e| e.step);
        assert!(first0.is_some(), "neuron 0 should spike");
        assert!(first1.is_some(), "neuron 1 should spike via the synapse");
        assert!(
            first1.unwrap() >= first0.unwrap() + net.params().delay_steps() as u64,
            "neuron 1 spiked too early: {:?} vs {:?} + {}",
            first1,
            first0,
            net.params().delay_steps()
        );
    }

    #[test]
    fn simulation_is_deterministic() {
        let build = || {
            let syn = Synapses::from_edges(3, &[0, 1], &[1, 2], &[0.01, 0.01], 18);
            let mut net = Network::new(3, syn);
            net.add_input(PoissonInput::new(vec![0], 150.0, net.params(), 2024));
            net
        };
        let mut a = build();
        let mut b = build();
        a.run(1.0);
        b.run(1.0);
        assert_eq!(a.spikes(), b.spikes());
        assert!(a.spike_count() > 0);
    }

    #[test]
    fn rate_query_matches_recorded_spikes() {
        let mut net = isolated(1, 0);
        net.add_input(PoissonInput::new(vec![0], 150.0, net.params(), 5));
        net.run(2.0);
        let n = net.spikes().iter().filter(|e| e.neuron == 0).count();
        let expected = n as f64 / 2.0;
        assert!((net.rate_hz(0) - expected).abs() < 1e-9);
    }
}
