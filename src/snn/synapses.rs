//! [`Synapses`] — sparse (CSR) connectivity plus a delay ring buffer that
//! delivers each presynaptic spike to its postsynaptic targets exactly `t_dly`
//! later.
//!
//! ## Connectivity (from `model.py:177-183`)
//!
//! `syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly)` with
//! `w = ExcXConn · w_syn`. So on a presynaptic spike, every outgoing synapse
//! adds its weight `w` to the postsynaptic neuron's `g` variable, after the
//! fixed transmission delay `t_dly`.
//!
//! ## Storage
//!
//! Connections are stored in compressed-sparse-row form keyed by the
//! **presynaptic** neuron: for source `i`, its outgoing edges are
//! `targets[row_start[i]..row_start[i+1]]` with matching `weights`. This makes
//! "for each neuron that spiked, fan out to its targets" a cache-friendly scan.
//!
//! ## Delay
//!
//! A ring buffer of `delay_steps + 1` slots accumulates *pending* per-neuron `g`
//! increments. When neuron `j`'s synapse fires at step `s`, the increment is
//! added to the slot for step `s + delay_steps`. Each step, the network drains
//! the current slot and applies it, then clears it.

/// Sparse CSR synapse table with a fixed-delay delivery buffer.
#[derive(Debug, Clone)]
pub struct Synapses {
    /// Number of neurons (rows).
    n: usize,
    /// CSR row offsets, length `n + 1`. Outgoing edges of neuron `i` are the
    /// half-open range `row_start[i]..row_start[i+1]`.
    row_start: Vec<usize>,
    /// Postsynaptic target index per edge.
    targets: Vec<u32>,
    /// Weight (V) per edge, already scaled by `w_syn`.
    weights: Vec<f64>,
    /// Delay ring buffer: `delay_slots` buffers of length `n`, each holding
    /// pending `g` increments to apply on a future step.
    ring: Vec<Vec<f64>>,
    /// Number of ring slots = `delay_steps + 1`.
    delay_slots: usize,
    /// The delay in whole steps.
    delay_steps: usize,
}

impl Synapses {
    /// Build a synapse table from parallel edge arrays (presynaptic index,
    /// postsynaptic index, weight) over `n` neurons, with a transmission delay
    /// of `delay_steps` timesteps.
    ///
    /// Edges may be given in any order; they are grouped by source into CSR.
    pub fn from_edges(
        n: usize,
        pre: &[u32],
        post: &[u32],
        weight: &[f64],
        delay_steps: usize,
    ) -> Self {
        assert_eq!(pre.len(), post.len());
        assert_eq!(pre.len(), weight.len());

        // Count out-degree per source, then prefix-sum into row_start.
        let mut row_start = vec![0usize; n + 1];
        for &i in pre {
            row_start[i as usize + 1] += 1;
        }
        for i in 0..n {
            row_start[i + 1] += row_start[i];
        }

        // Fill targets/weights using a moving cursor per row.
        let mut cursor = row_start.clone();
        let mut targets = vec![0u32; pre.len()];
        let mut weights = vec![0.0f64; pre.len()];
        for e in 0..pre.len() {
            let src = pre[e] as usize;
            let slot = cursor[src];
            targets[slot] = post[e];
            weights[slot] = weight[e];
            cursor[src] += 1;
        }

        let delay_slots = delay_steps + 1;
        let ring = vec![vec![0.0f64; n]; delay_slots];

        Self {
            n,
            row_start,
            targets,
            weights,
            ring,
            delay_slots,
            delay_steps,
        }
    }

    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no neurons.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Total number of directed synapses (edges).
    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }

    /// The configured delay in whole steps.
    pub fn delay_steps(&self) -> usize {
        self.delay_steps
    }

    /// Schedule delivery of all outgoing synapses of a neuron that spiked at
    /// `current_step`: each target's pending `g` increment lands `delay_steps`
    /// steps in the future.
    pub fn fan_out_spike(&mut self, source: usize, current_step: u64) {
        let future = (current_step as usize + self.delay_steps) % self.delay_slots;
        let start = self.row_start[source];
        let end = self.row_start[source + 1];
        // Split borrow: read edges, write into the chosen ring slot.
        let slot = &mut self.ring[future];
        for e in start..end {
            slot[self.targets[e] as usize] += self.weights[e];
        }
    }

    /// Take (and clear) the pending per-neuron `g` increments scheduled for
    /// `current_step`. Returns a slice the caller applies to the neurons.
    ///
    /// The returned buffer is owned by the ring; it is zeroed after the caller
    /// applies it via [`Synapses::clear_delivery_slot`].
    pub fn delivery_slot(&self, current_step: u64) -> &[f64] {
        let idx = current_step as usize % self.delay_slots;
        &self.ring[idx]
    }

    /// Zero the ring slot for `current_step` after its increments were applied,
    /// so the slot can be reused `delay_slots` steps later.
    pub fn clear_delivery_slot(&mut self, current_step: u64) {
        let idx = current_step as usize % self.delay_slots;
        for x in self.ring[idx].iter_mut() {
            *x = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_groups_edges_by_source() {
        // 3 neurons; edges 0->1 (w=1), 0->2 (w=2), 2->1 (w=3). No delay.
        let s = Synapses::from_edges(3, &[0, 0, 2], &[1, 2, 1], &[1.0, 2.0, 3.0], 0);
        assert_eq!(s.len(), 3);
        assert_eq!(s.edge_count(), 3);
    }

    #[test]
    fn zero_delay_delivers_same_step() {
        // delay_steps = 0 → a spike scheduled at step s is delivered at step s.
        let mut s = Synapses::from_edges(2, &[0], &[1], &[0.5], 0);
        s.fan_out_spike(0, 0);
        let slot = s.delivery_slot(0);
        assert_eq!(slot[1], 0.5);
        assert_eq!(slot[0], 0.0);
    }

    #[test]
    fn delay_delivers_exactly_n_steps_later() {
        let delay = 18usize;
        let mut s = Synapses::from_edges(2, &[0], &[1], &[0.5], delay);
        s.fan_out_spike(0, 0);
        // Nothing before the delay.
        for step in 0..delay as u64 {
            assert_eq!(s.delivery_slot(step)[1], 0.0, "early delivery at {step}");
        }
        // Delivered exactly at step `delay`.
        assert_eq!(s.delivery_slot(delay as u64)[1], 0.5);
    }

    #[test]
    fn cleared_slot_is_reusable() {
        let mut s = Synapses::from_edges(2, &[0], &[1], &[0.5], 1);
        s.fan_out_spike(0, 0); // delivers at step 1
        assert_eq!(s.delivery_slot(1)[1], 0.5);
        s.clear_delivery_slot(1);
        assert_eq!(s.delivery_slot(1)[1], 0.0);
    }
}
