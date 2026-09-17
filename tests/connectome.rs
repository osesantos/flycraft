//! Full-connectome integration tests: loader invariants (fast, run by default)
//! and full-network experiment validation (release-only, `cargo test --release
//! -- --ignored`).
//!
//! ## Reference numbers (for the sugar band checks)
//!
//! From fly-brain's own runs (`data/results/nature_2026_07/manifest.csv`,
//! Sugar GRNs (200 Hz), Brian2 (CPU) `brian2cpp`, t_run=0.1 s, n_run=1,
//! rounds 1–5): total spikes 1462–1574 (mean ≈1534), active neurons 307–334
//! (mean ≈323). Brian2's RNG is unseeded, so the reference itself varies;
//! the band here expands the observed min/max by ±10%.
//!
//! P9, sugar+P9 (`neu_exc2`) and silencing (`neu_slnc`) have no manifest entry,
//! so they are validated against a Brian2 oracle run locally with the same
//! stimulus config (reference `model.py` semantics, numpy target).

use flycraft::connectome::{Connectome, Experiment, P9, SUGAR, SUGAR_AND_P9};

fn connectome() -> Connectome {
    Connectome::load_repo_defaults().expect("v783 data shipped in flycraft/data")
}

#[test]
fn roster_has_every_v783_neuron() {
    let c = connectome();
    assert_eq!(c.neuron_count(), 138_639);
    assert_eq!(c.completeness().flyids()[0], 720575940596125868);
}

#[test]
fn flyid_index_roundtrips() {
    let c = connectome();
    for &f in &SUGAR.excited_flyids[..3] {
        let idx = c.completeness().idx_of(f).expect("roster has flyid");
        assert_eq!(
            c.completeness().flyid_of(idx),
            Some(f),
            "idx_of/flyid_of invert"
        );
    }
}

#[test]
fn experiment_flyids_resolve_to_roster() {
    let c = connectome();
    for exp in [SUGAR, P9, SUGAR_AND_P9] {
        let all = exp
            .excited_flyids
            .iter()
            .chain(exp.excited2_flyids)
            .chain(exp.silenced_flyids);
        for &f in all {
            assert!(
                c.completeness().idx_of(f).is_some(),
                "{} flyid {f} missing from roster",
                exp.name
            );
        }
    }
}

#[test]
fn edge_list_matches_file_invariants() {
    let c = connectome();
    // Pin exact row count observed in the v783 parquet.
    assert_eq!(c.edge_count(), 15_091_983);
    assert_eq!(c.connectivity().edge_count(), c.connectivity().pre().len());
    assert!(c.connectivity().post().len() == c.connectivity().edge_count());
    assert!(c.connectivity().weight().len() == c.connectivity().edge_count());
    let max = c.connectivity().max_index().expect("non-empty edge list");
    assert!(
        (max as usize) < c.neuron_count(),
        "edge index {max} outside {}-neuron roster",
        c.neuron_count()
    );
    // Weight envelope matches the file's `Excitatory x Connectivity` column
    // (range −2405…+1897, signed: −1 = inhibitory) scaled by w_syn.
    let w = c.connectivity().weight();
    let min_w = w.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_w = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let neg = w.iter().filter(|&&x| x < 0.0).count();
    assert!(
        (min_w - (-2405.0 * 0.000_275)).abs() <= 1e-12,
        "min weight {min_w}"
    );
    assert!(
        (max_w - (1897.0 * 0.000_275)).abs() <= 1e-12,
        "max weight {max_w}"
    );
    assert_eq!(neg, 6_032_681, "inhibitory (negative-weight) edge count");
}

/// Run an experiment on an already-loaded connectome and return
/// `(spikes, active_neurons, seconds)`.
fn aggregate_with(
    c: &Connectome,
    exp: Experiment,
    duration: f64,
    seed: u64,
) -> (usize, usize, f64) {
    let t = std::time::Instant::now();
    let mut net = c.network(&exp, seed).expect("network builds");
    net.run(duration);
    (
        net.spike_count(),
        net.active_neuron_count(),
        t.elapsed().as_secs_f64(),
    )
}

/// Load the connectome and run an experiment over `duration` seconds.
fn aggregate(exp: Experiment, duration: f64, seed: u64) -> (usize, usize, f64) {
    aggregate_with(&connectome(), exp, duration, seed)
}

/// Mean spike count over the given seeds on one loaded connectome.
fn mean_spikes(c: &Connectome, exp: Experiment, duration: f64, seeds: &[u64]) -> f64 {
    let total: usize = seeds
        .iter()
        .map(|&seed| aggregate_with(c, exp, duration, seed).0)
        .sum();
    total as f64 / seeds.len() as f64
}

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn sugar_t01_matches_reference_band() {
    let (spikes, active, secs) = aggregate(SUGAR, 0.1, 42);
    // Reference (brian2cpp, 5 rounds): spikes 1316–1731, active 276–367 after ±10%.
    println!("sugar t=0.1s: {spikes} spikes / {active} active in {secs:.1}s");
    assert!(
        (1316..=1731).contains(&spikes),
        "sugar spike count {spikes} outside reference band 1462–1574 (±10%)"
    );
    assert!(
        (276..=367).contains(&active),
        "sugar active-neuron count {active} outside reference band 307–334 (±10%)"
    );
}

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn sugar_seeded_run_is_deterministic() {
    let (a, _, _) = aggregate(SUGAR, 0.1, 7);
    let (b, _, _) = aggregate(SUGAR, 0.1, 7);
    assert_eq!(a, b, "same seed must reproduce the same spike count");
}

/// P9 excited at 100 Hz with its own outgoing synapses silenced (`neu_slnc`).
const P9_SILENCED: Experiment = Experiment {
    name: "p9+slnc",
    excited_flyids: P9.excited_flyids,
    stim_rate_hz: 100.0,
    excited2_flyids: &[],
    stim_rate2_hz: 0.0,
    silenced_flyids: P9.excited_flyids,
};

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn p9_matches_oracle() {
    // P9 drives only two neurons, so the downstream cascade is small and
    // high-variance: the Brian2 oracle (numpy, unseeded, 20 runs) spans 24–195
    // spikes with mean ≈72. Compare the mean over seeds, not a single run.
    let c = connectome();
    let mean = mean_spikes(&c, P9, 0.1, &[1, 2, 3, 4, 5]);
    println!("p9 t=0.1s: mean {mean:.0} spikes over 5 seeds (oracle mean ≈72, range 24–195)");
    assert!(
        (35.0..=130.0).contains(&mean),
        "p9 mean {mean} outside the oracle's observed range"
    );
}

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn sugar_and_p9_coactivation_drives_both_channels() {
    // `neu_exc` (sugar, 200 Hz) + `neu_exc2` (P9, 100 Hz). Oracle: 1616 spikes /
    // 335 active; sugar alone is ≈1534, so co-activation adds a modest tail.
    let c = connectome();
    let mean = mean_spikes(&c, SUGAR_AND_P9, 0.1, &[1, 2, 3, 4, 5]);
    println!("sugar+p9 t=0.1s: mean {mean:.0} spikes over 5 seeds (oracle 1616)");
    assert!(
        (1350.0..=1900.0).contains(&mean),
        "sugar+p9 mean {mean} far from the oracle's 1616"
    );
}

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn p9_silencing_suppresses_downstream() {
    let (loud, _, _) = aggregate(P9, 0.1, 42);
    let (quiet, active, _) = aggregate(P9_SILENCED, 0.1, 42);
    // Brian2 oracle: unsilenced 65 / 36 (single run); silenced 17 / 2.
    println!(
        "p9 silenced: {quiet} spikes / {active} active vs unsilenced {loud} (oracle 17/2 vs 65/36)"
    );
    assert!(quiet < loud, "silencing P9 must reduce network spikes");
    assert_eq!(active, 2, "with P9 muted only the two driven neurons fire");
    assert!(
        (5..=40).contains(&quiet),
        "silenced P9 should fire only its own Poisson events, got {quiet}"
    );
}
