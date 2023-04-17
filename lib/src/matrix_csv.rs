use color_eyre::eyre::{eyre, Report, Result};
use csv;
use nalgebra::base::{DMatrix, Scalar};

use std::path::Path;
use std::str::FromStr;

/// Returns DMatrix with target in the first position, and the
/// placeholder for intercept in the last position.
pub fn from_csv<P: AsRef<Path>, N>(path: P, with_headers: bool) -> Result<DMatrix<N>>
where
    N: FromStr + Scalar,
    <N as FromStr>::Err: Send + Sync,
    <N as FromStr>::Err: std::error::Error,
{
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(with_headers)
        .from_path(path)?;

    // stage the data on zero copy for Matrix
    let bias_slot = vec![&[b'1']];
    //
    let staged_records: Vec<N> = reader
        .byte_records()
        .map(|record| -> Result<Vec<N>> {
            let record: Vec<N> = record?
                .iter()
                .chain(bias_slot.iter().map(|&v| v.as_ref()))
                .map(|value| {
                    std::str::from_utf8(value)
                        .map_err(|e| eyre!("Error decoding utf8: {}", e))
                        .and_then(|s| s.parse().map_err(Report::from))
                })
                .collect::<Result<Vec<N>, _>>()?;
            Ok(record)
        })
        .collect::<Result<Vec<Vec<N>>>>()?
        .into_iter()
        .flatten()
        .collect();

    // feature count + intercept slot
    let feature_count = reader.headers()?.len() + 1;
    let num_records = staged_records.len() / feature_count;

    let staged_records: &[N] = &staged_records;
    let dmatrix = DMatrix::from_row_slice(num_records, feature_count, staged_records);

    Ok(dmatrix)
}
