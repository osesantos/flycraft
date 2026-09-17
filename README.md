# FlyCraft

**Teaching a fruit-fly brain to fly a drone.**

FlyCraft is a native-Rust engine for a long-running research question: *can a
connectome-derived* Drosophila *brain fly a drone?* The reference brain model,
[fly-brain](../fly-brain), is a whole-brain leaky integrate-and-fire (LIF)
network built from the FlyWire connectome (~138k neurons, ~15M synapses; Shiu et
al.). It is excellent science but GPL Python, CUDA-bound, and batch/offline —
each simulated step costs seconds, far too slow to close a real-time control
loop.

FlyCraft reimplements the **same LIF model natively in Rust** so the network can
be *built once and stepped repeatedly*, MIT-clean and dependency-light.

## Milestone 1 — the LIF engine

The `snn` module is a faithful, deterministic Rust port of fly-brain's neuron
model (`code/paper-phil-drosophila/model.py`). It is validated **offline against
the source equations** on small known-answer networks — no drone, no real-time,
no Python yet.

What it contains:

| Module | Responsibility |
|--------|----------------|
| `snn::params` | `LifParams` — every constant, mapped to its `model.py` source line |
| `snn::neuron` | Per-neuron `(v, g)` state; **exact linear** integrator (matches Brian2's `method='linear'`, not Euler); threshold, reset, refractory |
| `snn::synapses` | Sparse CSR connectivity + delay ring buffer (spike bumps postsynaptic `g` exactly `t_dly` later) |
| `snn::poisson` | Deterministic seeded Poisson input (hand-rolled SplitMix64 PRNG, no external crate) driving membrane `v` |
| `snn::network` | `Network::step`/`run`, spike recording, firing-rate query |

### Fidelity basis

Every equation and constant comes from fly-brain's `model.py`. The within-step
ordering replicates Brian2 exactly: **integrate + threshold + reset (fanned out
after `t_dly`) → deliver delayed synaptic events → apply Poisson input**. Delivery
runs *after* the state update, and events landing on a neuron that spiked or is
refractory this step are discarded — matching Brian2's reset-on-spike and frozen
`g` during the refractory period. The integrator uses the closed-form solution of
the coupled linear ODEs (verified against a 1 ns-resolution Euler reference in the
tests), so there is no discretisation error versus the reference model.

Poisson input is deterministic under a fixed seed. Exact spike-by-spike parity
with Brian2's own random stream is *not* a goal of M1; the engine is validated
against the model's **equations** and against fly-brain's coarse published
aggregates (spike counts / active-neuron counts in its run manifest).

### Example

```rust
use flycraft::snn::{Network, LifParams};
use flycraft::snn::synapses::Synapses;
use flycraft::snn::poisson::PoissonInput;

let params = LifParams::fly_brain();

// One neuron, no synapses, driven at 150 Hz Poisson (the model default r_poi).
let synapses = Synapses::from_edges(1, &[], &[], &[], params.delay_steps());
let mut net = Network::with_params(params, 1, synapses);
net.add_input(PoissonInput::new(vec![0], 150.0, net.params(), /*seed=*/42));

net.run(1.0); // simulate one second
println!("fired {} spikes ({:.1} Hz)", net.spike_count(), net.rate_hz(0));
```

## Milestone 2 — the connectome loader (this milestone)

The `connectome` module loads the real FlyWire **v783** male-CNS connectome and
wires it straight into the M1 engine:

| Module | Responsibility |
|--------|----------------|
| `connectome::completeness` | `2025_Completeness_783.csv` → the 138,639-neuron roster and flyid↔index mapping |
| `connectome::connectivity` | `2025_Connectivity_783.parquet` → 15,091,983 signed, weighted edges (`Excitatory x Connectivity x w_syn`) |
| `connectome::experiment` | Stimulus definitions (`SUGAR`, `P9`, `SUGAR_AND_P9`) and their target flyids |

Both data files live in `data/` (git-ignored; copy them from
[fly-brain](../fly-brain)'s `data/` directory). The parquet is read natively through
`arrow`/`parquet` — no Python — with weights taken as `Excitatory x Connectivity *
w_syn` and indices mapped by CSV row order, exactly as the reference does.

Each `Experiment` mirrors the reference's full stimulus vocabulary
(`model.py:61-127`): `neu_exc` neurons are Poisson-driven at `stim_rate_hz`
(`r_poi`), `neu_exc2` at a second rate (`r_poi2`), and `neu_slnc` neurons have every
outgoing synapse weight zeroed — silencing a neuron's *influence*, not its own
spiking. So arbitrary sensory groups can be prodded without touching the engine.

### Validation

The `sugar` experiment (21 gustatory-receptor neurons driven at 200 Hz) is checked
against fly-brain's run manifest, and the other stimulus channels against a Brian2
oracle run locally with matching configs:

| Quantity (t = 0.1 s) | FlyCraft | Reference |
|----------------------|----------|-----------|
| sugar, spikes (seed 42) | 1551 | 1462–1574 (mean 1534) |
| sugar, spikes (10-seed mean) | 1532 | 1534 |
| sugar, active neurons (10-seed mean) | 324 | 323 |
| sugar + P9 (`neu_exc2`), spikes (5-seed mean) | 1617 | 1616 (oracle) |
| P9, spikes (5-seed mean) | 59 | 72 (oracle, range 24–195) |
| P9 silenced (`neu_slnc`), spikes / active | 21 / 2 | 17 / 2 (oracle) |

At t = 1 s FlyCraft produces 17,101 spikes against the reference oracle's 17,034
(0.4%). Two subtleties of Brian2's step ordering were needed for parity: delayed
events are delivered *after* the state update, and events landing on a spiking or
refractory neuron are discarded. P9 is a small, high-variance cascade (the oracle
itself spans 24–195 spikes), so it is validated on a seed mean rather than one run.

## Roadmap

- **M1 — LIF engine** (done): numerically-faithful, deterministic, offline.
- **M2 — Connectome loader** (done): the real v783 connectivity runs in the
  engine and reproduces fly-brain's `sugar` aggregate spike counts within ~1%.
- **M3+ — Closing the loop**: encode drone state into sensory neuron activity,
  decode descending-neuron spikes into actuator commands, and fly.

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Licence

MIT. fly-brain (the reference model) is GPL-2.0-or-later; FlyCraft is a clean
reimplementation of the *model*, not a copy of its code, and links none of it.
