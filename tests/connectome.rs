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

use flycraft::connectome::{Connectome, Experiment, P9, SUGAR};

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
    for (i, exp) in [SUGAR, P9].into_iter().enumerate() {
        assert!(!exp.excited_flyids.is_empty());
        for &f in exp.excited_flyids {
            assert!(
                c.completeness().idx_of(f).is_some(),
                "{} flyid {f} missing from roster",
                exp.name
            );
        }
        let _ = i;
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

/// Run a full-network experiment and return `(spikes, active_neurons, seconds)`.
fn aggregate(exp: Experiment, duration: f64, seed: u64) -> (usize, usize, f64) {
    let c = connectome();
    let t = std::time::Instant::now();
    let mut net = c.network(&exp, seed).expect("network builds");
    net.run(duration);
    let secs = t.elapsed().as_secs_f64();
    (net.spike_count(), net.active_neuron_count(), secs)
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

#[test]
#[ignore = "full-network run; use: cargo test --release -- --ignored"]
fn p9_smoke_drives_the_descending_network() {
    let (spikes, active, secs) = aggregate(P9, 0.1, 42);
    println!("p9 t=0.1s: {spikes} spikes / {active} active in {secs:.1}s (no reference)");
    assert!(spikes > 0, "P9 stimulation must produce spikes");
    let (again, _, _) = aggregate(P9, 0.1, 42);
    assert_eq!(
        spikes, again,
        "same seed must reproduce the same spike count"
    );
}
