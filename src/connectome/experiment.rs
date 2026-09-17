/// An experiment: which neurons to excite, and at what rate.
///
/// Faithful copies of the `benchmark.py:57-99` definitions (same flyids, same
/// `stim_rate`, which becomes the Poisson rate `r_poi` via
/// `run_brian2_cuda.py:372`).
#[derive(Debug, Clone, Copy)]
pub struct Experiment {
    /// Experiment key, e.g. `"sugar"`.
    pub name: &'static str,
    /// FlyWire flyids of the excited (Poisson-driven) neurons.
    pub excited_flyids: &'static [u64],
    /// Stimulation rate (Hz) of the `neu_exc` Poisson inputs.
    pub stim_rate_hz: f64,
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
};
