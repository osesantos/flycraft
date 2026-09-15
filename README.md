# FlyCraft

**Teaching a fruit-fly brain to fly a drone.**

FlyCraft is a native-Rust engine for a long-running research question: *can a
connectome-derived* Drosophila *brain fly a drone?* The reference brain model,
[fly-brain](../fly-brain), is a whole-brain leaky integrate-and-fire (LIF)
network built from the FlyWire connectome (~138k neurons, ~5M synapses; Shiu et
al.). It is excellent science but GPL Python, CUDA-bound, and batch/offline —
each simulated step costs seconds, far too slow to close a real-time control
loop.

FlyCraft reimplements the **same LIF model natively in Rust** so the network can
be *built once and stepped repeatedly*, MIT-clean and dependency-light.

## Milestone 1 — the LIF engine (this milestone)

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
ordering replicates Brian2: **deliver delayed synaptic events → apply Poisson
input → integrate + threshold → reset → fan out spikes**. The integrator uses
the closed-form solution of the coupled linear ODEs (verified against a
1 ns-resolution Euler reference in the tests), so there is no discretisation
error versus the reference model.

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

## Roadmap

- **M1 — LIF engine** (done): numerically-faithful, deterministic, offline.
- **M2 — Connectome loader**: load the real FlyWire v783 connectivity
  (`2025_Completeness_783.csv`, `2025_Connectivity_783.parquet`) into the engine
  and reproduce a full-network experiment (e.g. `p9`), checked against
  fly-brain's aggregate outputs.
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
