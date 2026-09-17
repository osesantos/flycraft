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
//! - Experiment flavour matches `benchmark.py:57-99`
//!   (`neu_exc`, `stim_rate` → `r_poi`). Excitatory neurons get Poisson on `v`
//!   at `w_syn·f_poi`, refractory disabled — already honoured by `Network`.

mod completeness;
mod connectivity;
mod experiment;

pub use completeness::Completeness;
pub use connectivity::Connectivity;
pub use experiment::{Experiment, P9, SUGAR};

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

    /// Build a full, at-rest [`Network`] for `experiment`, driven by seeded
    /// Poisson input. Same seed ⇒ identical simulation.
    pub fn network(
        &self,
        experiment: &Experiment,
        seed: u64,
    ) -> Result<Network, Box<dyn std::error::Error>> {
        let n = self.neuron_count();
        let synapses = Synapses::from_edges(
            n,
            self.connectivity.pre(),
            self.connectivity.post(),
            self.connectivity.weight(),
            self.params.delay_steps(),
        );
        let mut net = Network::with_params(self.params, n, synapses);
        let excited: Vec<u32> = experiment
            .excited_flyids
            .iter()
            .map(|flyid| {
                self.completeness.idx_of(*flyid).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "flyid {flyid} (experiment {}) not in roster",
                            experiment.name
                        ),
                    )
                })
            })
            .collect::<io::Result<_>>()?;
        net.add_input(PoissonInput::new(
            excited,
            experiment.stim_rate_hz,
            &self.params,
            seed,
        ));
        Ok(net)
    }
}
