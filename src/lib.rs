//! FlyCraft — a native-Rust *Drosophila* fly-brain engine.
//!
//! The long-term goal of FlyCraft is to explore whether a connectome-derived
//! fruit-fly brain can fly a drone. The `fly-brain` reference implementation is
//! GPL Python, CUDA-bound, and batch/offline (~1–4 s per simulated step) — far
//! too slow to close a real-time control loop. FlyCraft reimplements the same
//! leaky integrate-and-fire (LIF) model natively in Rust so the network can be
//! **built once and stepped repeatedly**, MIT-clean and dependency-light.
//!
//! ## Milestone 1 — the LIF engine
//!
//! This crate contains [`snn`]: a faithful, deterministic Rust port of
//! the Shiu et al. LIF neuron model used by `fly-brain`
//! (`code/paper-phil-drosophila/model.py`). M1 proves the engine is
//! **numerically correct** against that model's source equations on small
//! known-answer networks — no drone, no real-time, no Python.
//!
//! ## Milestone 2 — the connectome loader
//!
//! [`connectome`] loads the real FlyWire v783 connectome (138,639 neurons,
//! 15,091,983 synapses) shipped in `data/` into the M1 engine and reproduces
//! full-network experiments (sugar, P9) validated against fly-brain's reference
//! aggregates.

pub mod connectome;
pub mod snn;
