//! `snn` — a native-Rust spiking neural network engine reproducing the
//! *Drosophila* whole-brain leaky integrate-and-fire (LIF) model from
//! `fly-brain` (`code/paper-phil-drosophila/model.py`, Shiu et al.).
//!
//! ## Design
//!
//! - [`params::LifParams`] — every constant, mapped to its source line.
//! - [`neuron`] — per-neuron `(v, g)` state with the **exact linear** integrator
//!   (matching Brian2's `method='linear'`, not Euler), plus threshold, reset,
//!   and refractory handling.
//! - [`synapses::Synapses`] — sparse CSR connectivity with a delay ring buffer so
//!   a presynaptic spike bumps postsynaptic `g` exactly `t_dly` later.
//! - [`poisson`] — deterministic, seeded Poisson input (hand-rolled SplitMix64
//!   PRNG, no external crate) driving membrane voltage `v` on excited neurons.
//! - [`network::Network`] — owns all of the above; `step`/`run` advance the
//!   simulation and record spikes.
//!
//! ## Fidelity basis
//!
//! Every equation and constant is taken from `fly-brain`'s `model.py`. The
//! within-step ordering replicates Brian2: integrate → threshold → reset →
//! deliver delayed synaptic events. Poisson input is deterministic under a fixed
//! seed; exact spike-by-spike parity with Brian2's RNG is out of scope (M1
//! validates against the *equations* and coarse published aggregates).

pub mod network;
pub mod neuron;
pub mod params;
pub mod poisson;
pub mod synapses;

pub use network::Network;
pub use params::LifParams;
