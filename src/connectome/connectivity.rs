use arrow::array::{as_primitive_array, Array, Int64Array};
use arrow::datatypes::Int64Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::snn::params::LifParams;

/// The v783 directed edge list: every synapse in the connectome.
///
/// Loaded from `2025_Connectivity_783.parquet` (15,091,983 rows). The reference
/// model connects one synapse per row and weights it `w = ExcXConn · w_syn`
/// (`model.py:160-183`), where `Excitatory x Connectivity` is an integer synapse
/// multiplicity (1–1897 in this file). Multiplying by `w_syn` here gives the
/// engine one weighted edge per row — the same total postsynaptic `g` increment
/// per presynaptic spike as the reference.
#[derive(Debug, Clone)]
pub struct Connectivity {
    pre: Vec<u32>,
    post: Vec<u32>,
    weight: Vec<f64>,
}

impl Connectivity {
    /// Load every edge from the v783 connectivity parquet.
    ///
    /// Uses the `Presynaptic_Index` / `Postsynaptic_Index` local indices
    /// already present in the file (Indices refer to rows of the completeness
    /// roster). Weights are `Excitatory x Connectivity · w_syn`.
    pub fn from_parquet(
        path: &Path,
        params: &LifParams,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let rows = builder.metadata().file_metadata().num_rows() as usize;
        let mut reader = builder.build()?;

        let mut pre = Vec::with_capacity(rows);
        let mut post = Vec::with_capacity(rows);
        let mut weight = Vec::with_capacity(rows);

        while let Some(batch) = reader.next().transpose()? {
            let (p, po, w) = read_batch(&batch, params)?;
            pre.extend_from_slice(&p);
            post.extend_from_slice(&po);
            weight.extend_from_slice(&w);
        }
        Ok(Self { pre, post, weight })
    }

    /// Number of directed synapses (edges).
    pub fn edge_count(&self) -> usize {
        self.pre.len()
    }

    /// Presynaptic local index per edge.
    pub fn pre(&self) -> &[u32] {
        &self.pre
    }

    /// Postsynaptic local index per edge.
    pub fn post(&self) -> &[u32] {
        &self.post
    }

    /// Synaptic weight (V) per edge, already scaled by `w_syn`.
    pub fn weight(&self) -> &[f64] {
        &self.weight
    }

    /// Largest presynaptic or postsynaptic index present in the edge list, if any.
    pub fn max_index(&self) -> Option<u32> {
        self.pre.iter().chain(self.post.iter()).max().copied()
    }
}

/// Parallel edge arrays read from one parquet record batch.
type EdgeBatch = (Vec<u32>, Vec<u32>, Vec<f64>);

fn read_batch(
    batch: &arrow::array::RecordBatch,
    params: &LifParams,
) -> Result<EdgeBatch, Box<dyn std::error::Error>> {
    let pre = i64_col(batch, "Presynaptic_Index")?;
    let post = i64_col(batch, "Postsynaptic_Index")?;
    let excx = i64_col(batch, "Excitatory x Connectivity")?;
    let n = pre.len();
    let mut p = Vec::with_capacity(n);
    let mut po = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        p.push(u32::try_from(pre.value(i))?);
        po.push(u32::try_from(post.value(i))?);
        w.push(excx.value(i) as f64 * params.w_syn);
    }
    Ok((p, po, w))
}

fn i64_col<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &str,
) -> Result<&'a Int64Array, Box<dyn std::error::Error>> {
    let idx = batch.schema().index_of(name)?;
    let arr = as_primitive_array::<Int64Type>(batch.column(idx));
    if arr.null_count() > 0 {
        return Err(format!("column `{name}` contains nulls").into());
    }
    Ok(arr)
}
