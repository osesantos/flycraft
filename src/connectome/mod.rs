//! `connectome` — load the real FlyWire v783 connectome into the M1 engine.
//!
//! M1 proved the LIF math on tiny known-answer networks; this module proves the
//! **loader + engine at connectome scale**. It reads the two v783 files shipped
//! in `data/` — `2025_Completeness_783.csv` (138,639 neurons, row order = local
//! index) and `2025_Connectivity_783.parquet` (15,091,983 synapses) — and
//! assembles a ready-to-run [`Network`](crate::snn::Network) for an
//! [`Experiment`] (e.g. sugar, P9) through the M1 engine seams
//! ([`Synapses::from_edges`](crate::snn::synapses::Synapses::from_edges) +
//! [`PoissonInput`](crate::snn::poisson::PoissonInput)).
//!
//! ## Fidelity basis
//!
//! - Index semantics match `run_brian2_cuda.py:381`
//!   (`flyid2i = {j: i for i, j in enumerate(df_comp.index)}`).
//! - Edges and weights match `model.py:160-183`
//!   (`syn.connect(i=idx_pre, j=idx_post); syn.w = ExcXConn · w_syn`).
//! - Experiment flavour matches `benchmark.py:57-99` and `model.py:61-127`:
//!   `neu_exc` at `r_poi`, `neu_exc2` at `r_poi2`, `neu_slnc` silenced. Excitatory
//!   neurons get Poisson on `v` at `w_syn·f_poi` with refractory disabled;
//!   silenced neurons keep spiking but have every outgoing weight zeroed.

mod completeness;
mod connectivity;
mod experiment;

pub use completeness::Completeness;
pub use connectivity::Connectivity;
pub use experiment::{Experiment, P9, SUGAR, SUGAR_AND_P9};

use crate::snn::network::Network;
use crate::snn::params::LifParams;
use crate::snn::poisson::PoissonInput;
use crate::snn::synapses::Synapses;
use std::io;
use std::path::{Path, PathBuf};

/// A loaded connectome: roster + edge list + model parameters.
///
/// The assembled network is built through the M1 seams, so the loader is a pure
/// data-mapping layer; the engine itself is unchanged.
#[derive(Debug, Clone)]
pub struct Connectome {
    completeness: Completeness,
    connectivity: Connectivity,
    params: LifParams,
}

impl Connectome {
    /// Load both v783 data files from `data_dir` with the given parameters.
    pub fn load(data_dir: &Path, params: LifParams) -> Result<Self, Box<dyn std::error::Error>> {
        let completeness = Completeness::from_csv(&data_dir.join("2025_Completeness_783.csv"))?;
        let connectivity =
            Connectivity::from_parquet(&data_dir.join("2025_Connectivity_783.parquet"), &params)?;
        if let Some(m) = connectivity.max_index() {
            if m as usize >= completeness.len() {
                return Err(format!(
                    "edge index {m} outside the {}-neuron roster",
                    completeness.len()
                )
                .into());
            }
        }
        Ok(Self {
            completeness,
            connectivity,
            params,
        })
    }

    /// Load with default fly-brain parameters from the copy of the data shipped
    /// in this repository (`flycraft/data/`).
    pub fn load_repo_defaults() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
        Self::load(&dir, LifParams::fly_brain())
    }

    /// Number of neurons in the connectome.
    pub fn neuron_count(&self) -> usize {
        self.completeness.len()
    }

    /// Number of directed synapses.
    pub fn edge_count(&self) -> usize {
        self.connectivity.edge_count()
    }

    /// Browser over the neuron roster.
    pub fn completeness(&self) -> &Completeness {
        &self.completeness
    }

    /// Browser over the directed edge list.
    pub fn connectivity(&self) -> &Connectivity {
        &self.connectivity
    }

    /// Resolve experiment flyids to local neuron indices, erroring if any flyid
    /// is missing from the roster.
    fn resolve_indices(
        &self,
        flyids: &[u64],
        experiment_name: &str,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        flyids
            .iter()
            .map(|flyid| {
                self.completeness.idx_of(*flyid).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("flyid {flyid} (experiment {experiment_name}) not in roster"),
                    )
                    .into()
                })
            })
            .collect()
    }

    /// Build a full, at-rest [`Network`] for `experiment`, driven by seeded
    /// Poisson input. Same seed ⇒ identical simulation.
    pub fn network(
        &self,
        experiment: &Experiment,
        seed: u64,
    ) -> Result<Network, Box<dyn std::error::Error>> {
        let n = self.neuron_count();
        let silenced = self.resolve_indices(experiment.silenced_flyids, experiment.name)?;
        let silenced_weights;
        let weight: &[f64] = if silenced.is_empty() {
            self.connectivity.weight()
        } else {
            silenced_weights = silence_outgoing(
                self.connectivity.pre(),
                self.connectivity.weight(),
                &silenced,
                n,
            );
            &silenced_weights
        };
        let synapses = Synapses::from_edges(
            n,
            self.connectivity.pre(),
            self.connectivity.post(),
            weight,
            self.params.delay_steps(),
        );
        let mut net = Network::with_params(self.params, n, synapses);

        let excited = self.resolve_indices(experiment.excited_flyids, experiment.name)?;
        net.add_input(PoissonInput::new(
            excited,
            experiment.stim_rate_hz,
            &self.params,
            seed,
        ));

        if !experiment.excited2_flyids.is_empty() {
            let excited2 = self.resolve_indices(experiment.excited2_flyids, experiment.name)?;
            net.add_input(PoissonInput::new(
                excited2,
                experiment.stim_rate2_hz,
                &self.params,
                seed.wrapping_add(1),
            ));
        }
        Ok(net)
    }
}

/// Copy `weight`, zeroing every edge whose presynaptic neuron is in `silenced`.
/// Matches `model.py:111-127` (`syn.w['i == pre'] = 0`), which removes a
/// neuron's outgoing influence without touching its own dynamics.
fn silence_outgoing(pre: &[u32], weight: &[f64], silenced: &[u32], n: usize) -> Vec<f64> {
    let mut mask = vec![false; n];
    for &i in silenced {
        mask[i as usize] = true;
    }
    pre.iter()
        .zip(weight)
        .map(|(&p, &w)| if mask[p as usize] { 0.0 } else { w })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::silence_outgoing;

    #[test]
    fn silencing_zeroes_only_outgoing_edges() {
        // Edges 0->1 (1.0), 0->2 (2.0), 1->2 (3.0), 2->0 (4.0).
        let pre = [0u32, 0, 1, 2];
        let w = [1.0f64, 2.0, 3.0, 4.0];
        // Silencing neuron 0 zeroes its outgoing edges only.
        assert_eq!(
            silence_outgoing(&pre, &w, &[0], 3),
            vec![0.0, 0.0, 3.0, 4.0]
        );
        // Silencing neuron 2 zeroes only the edge it sources.
        assert_eq!(
            silence_outgoing(&pre, &w, &[2], 3),
            vec![1.0, 2.0, 3.0, 0.0]
        );
        // No silencing is a faithful copy.
        assert_eq!(silence_outgoing(&pre, &w, &[], 3), w.to_vec());
    }
}
