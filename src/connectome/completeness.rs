use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// The v783 neuron roster: FlyWire flyid per local index.
///
/// Loaded from `2025_Completeness_783.csv`. Its index column holds the FlyWire
/// flyid for each of the 138,639 neurons; the **CSV row order is the local
/// neuron index** (`flyid2i = {j: i for i, j in enumerate(df_comp.index)}`,
/// `run_brian2_cuda.py:381`). The `Completed` flag is ignored — the reference
/// model uses every row as a neuron (`N = len(df_comp)`, `model.py:164`).
#[derive(Debug, Clone)]
pub struct Completeness {
    flyids: Vec<u64>,
    idx: HashMap<u64, u32>,
}

impl Completeness {
    /// Parse the completeness CSV into the roster.
    pub fn from_csv(path: &Path) -> std::io::Result<Self> {
        let mut flyids = Vec::new();
        let mut idx = HashMap::new();
        for (lineno, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let line = line?;
            if lineno == 0 || line.trim().is_empty() {
                continue; // header `,Completed`
            }
            let flyid_raw = line
                .split(',')
                .next()
                .ok_or_else(|| invalid(&line, lineno, "missing flyid"))?
                .trim();
            let flyid: u64 = flyid_raw
                .parse()
                .map_err(|_| invalid(&line, lineno, "flyid is not an integer"))?;
            if idx.insert(flyid, flyids.len() as u32).is_some() {
                return Err(invalid(&line, lineno, "duplicate flyid"));
            }
            flyids.push(flyid);
        }
        Ok(Self { flyids, idx })
    }

    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.flyids.len()
    }

    /// Whether the roster is empty.
    pub fn is_empty(&self) -> bool {
        self.flyids.is_empty()
    }

    /// All flyids in local-index order (CSV row order).
    pub fn flyids(&self) -> &[u64] {
        &self.flyids
    }

    /// Local index of `flyid`, if present.
    pub fn idx_of(&self, flyid: u64) -> Option<u32> {
        self.idx.get(&flyid).copied()
    }

    /// Flyid at local index `idx`, if present.
    pub fn flyid_of(&self, idx: u32) -> Option<u64> {
        self.flyids.get(idx as usize).copied()
    }
}

fn invalid(line: &str, lineno: usize, why: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("line {}: {} (`{line}`)", lineno + 1, why),
    )
}
