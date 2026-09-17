/// An experiment: which neurons to excite (at one or two rates), and which to
/// silence.
///
/// Faithful to the reference's stimulus vocabulary (`model.py:61-127`):
/// `neu_exc` at `r_poi`, `neu_exc2` at `r_poi2`, and `neu_slnc` whose outgoing
/// synapses are zeroed. `benchmark.py:57-99` ships `sugar` and `p9`.
#[derive(Debug, Clone, Copy)]
pub struct Experiment {
    /// Experiment key, e.g. `"sugar"`.
    pub name: &'static str,
    /// `neu_exc`: flyids Poisson-driven at `stim_rate_hz`.
    pub excited_flyids: &'static [u64],
    /// Stimulation rate (Hz) of the `neu_exc` Poisson inputs.
    pub stim_rate_hz: f64,
    /// `neu_exc2`: flyids Poisson-driven at the second rate `stim_rate2_hz`
    /// (`r_poi2`). Empty for every shipped experiment.
    pub excited2_flyids: &'static [u64],
    /// Stimulation rate (Hz) of the `neu_exc2` inputs (`r_poi2`).
    pub stim_rate2_hz: f64,
    /// `neu_slnc`: flyids whose outgoing synapses are zeroed (`model.py:111-127`).
    /// Silencing removes a neuron's influence on its targets, not its own spiking.
    pub silenced_flyids: &'static [u64],
}

/// Sugar sensory stimulation: 21 gustatory receptor neurons at 200 Hz
/// (`EXPERIMENTS['sugar']`). Validation experiment with published aggregate
/// references (manifest.csv).
pub const SUGAR: Experiment = Experiment {
    name: "sugar",
    excited_flyids: &[
        720575940624963786,
        720575940630233916,
        720575940637568838,
        720575940638202345,
        720575940617000768,
        720575940630797113,
        720575940632889389,
        720575940621754367,
        720575940621502051,
        720575940640649691,
        720575940639332736,
        720575940616885538,
        720575940639198653,
        720575940639259967,
        720575940617937543,
        720575940632425919,
        720575940633143833,
        720575940612670570,
        720575940628853239,
        720575940629176663,
        720575940611875570,
    ],
    stim_rate_hz: 200.0,
    excited2_flyids: &[],
    stim_rate2_hz: 0.0,
    silenced_flyids: &[],
};

/// P9 descending-neuron stimulation: 2 command neurons at 100 Hz
/// (`EXPERIMENTS['p9']`). The natural motor hook for later milestones; no
/// aggregate reference exists in the manifest, so it serves as a wiring smoke
/// test.
pub const P9: Experiment = Experiment {
    name: "p9",
    excited_flyids: &[
        720575940627652358, // P9 left
        720575940635872101, // P9 right
    ],
    stim_rate_hz: 100.0,
    excited2_flyids: &[],
    stim_rate2_hz: 0.0,
    silenced_flyids: &[],
};

/// Co-activation: sugar GRNs at 200 Hz (`neu_exc`) plus the P9 pair at 100 Hz
/// (`neu_exc2`). Both flyid sets are reference-defined, but fly-brain publishes
/// no aggregate for the combination — it exercises the second stimulus channel.
pub const SUGAR_AND_P9: Experiment = Experiment {
    name: "sugar+p9",
    excited_flyids: SUGAR.excited_flyids,
    stim_rate_hz: 200.0,
    excited2_flyids: P9.excited_flyids,
    stim_rate2_hz: 100.0,
    silenced_flyids: &[],
};
