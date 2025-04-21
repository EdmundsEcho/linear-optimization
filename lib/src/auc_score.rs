use color_eyre::eyre::{eyre, Result};
use nalgebra::base::DVector;
use num_traits::Float;
use std::fmt::Display;
use std::iter::zip;
use std::ops::AddAssign;
use tracing::{event, Level};

// assess if the value is zero
#[derive(Debug)]
struct Count<T> {
    neg: T,
    pos: T,
}
pub fn auc_score<T>(y_true: &DVector<T>, y_hat: &DVector<T>) -> Result<f64>
where
    T: Float + AddAssign + Display,
{
    //
    // validate and count negative and positive values
    //
    let mut counts = Count {
        pos: T::zero(),
        neg: T::zero(),
    };

    let mut warn_msg: &str = "";
    y_true.iter().fold(&mut counts, |acc, yi| {
        if yi == &T::zero() {
            acc.neg += T::one();
        } else if yi == &T::one() {
            acc.pos += T::one();
        } else if yi > &T::one() {
            acc.pos += T::one();
            warn_msg = "🟡 Found at least one non-binary value.";
        } else {
            panic!(
                "binary quality score (auc): only for binary classification. Invalid label: {}",
                yi
            );
        }
        acc
    });

    if !warn_msg.is_empty() {
        event!(Level::INFO, "{}", &warn_msg);
    }

    let auc: usize = zip(y_true.iter(), y_hat.iter())
        .filter(|(yi_true, yi_hat)| yi_true == yi_hat)
        .count();

    let pos = counts
        .pos
        .to_f64()
        .take()
        .ok_or(eyre!("Failed to count 1"))?;
    let neg = counts
        .neg
        .to_f64()
        .take()
        .ok_or(eyre!("Failed to count 0"))?;

    // let result = (auc as f64 - (pos * (pos + 1f64) / 2f64)) / (pos * neg);
    let result = (auc as f64) / (pos + neg);

    Ok(result)
}
