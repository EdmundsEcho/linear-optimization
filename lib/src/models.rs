use color_eyre::eyre::Result;
use nalgebra::base::{DMatrix, DVector, Scalar};
use tracing::{event, Level};

use std::borrow::Borrow;
use std::cmp::PartialOrd;
use std::fmt;
use std::ops::Neg;
use std::path::Path;

use crate::auc_score::*;
use crate::matrix_csv;

///
/// specify the objective
/// x & y can be different versions.  they all need to something that casts to float.
///
pub struct Objective {
    pub x: DMatrix<f64>,
    pub y: DVector<f64>,
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Objective with x matrix {} x {} and y vector length {}",
            self.x.nrows(),
            self.x.ncols(),
            self.y.len()
        )
    }
}
impl fmt::Debug for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Objective")
            .field("x", &self.x.shape())
            .field("y", &self.y.len())
            .finish()
    }
}
impl<'a> Objective {
    pub fn new(x: DMatrix<f64>, y: DVector<f64>) -> Self {
        Objective { x, y }
    }
    pub fn from_csv<P: AsRef<Path>>(path: P, with_headers: bool) -> Result<Self> {
        // csv -> DMatrix<f64> with placeholder for intercept
        let matrix: DMatrix<f64> = matrix_csv::from_csv(path, with_headers)?;
        Ok(matrix.into())
    }
    /// The data has target in the first slot, and bias/intercept in the last slot
    pub fn from_vec(data: Vec<f64>, rows: usize) -> Result<Self> {
        // assert rows make sense given length
        let cols = data.len() / rows;
        let dmatrix = DMatrix::from_row_slice(rows, cols, &data);
        Ok(dmatrix.into())
    }
    ///
    /// Build using X separate from Y
    ///
    pub fn from_vecs(x: Vec<f64>, y: Vec<f64>, rows: usize) -> Result<Self> {
        // assert rows make sense given length
        let x_cols = x.len() / rows;
        let x_dmatrix = DMatrix::from_row_slice(rows, x_cols, &x);
        Ok(Objective::new(x_dmatrix, y.into()))
    }
    /// alias that points to trait
    pub fn from_matrix(matrix: DMatrix<f64>) -> Result<Self> {
        Ok(matrix.into())
    }
    pub fn feature_count(&self) -> usize {
        self.x.shape().1
    }
}

/// The logit target must be in the first column of the matrix.
impl std::convert::From<DMatrix<f64>> for Objective {
    fn from(matrix: DMatrix<f64>) -> Self {
        let (_, w) = matrix.shape();
        let y: DVector<f64> = matrix.column(0).into();
        let x: DMatrix<f64> = matrix.columns(1, w - 1).into();
        // todo: make this assertion more in-line with binary the assertion
        assert!(y.min() == 0.0 && y.max() == 1.0);
        Objective { x, y }
    }
}

#[derive(Debug)]
///
/// Findings from an optimization.  There is a dependency between all_betas and the
/// objective used to derive it.  The dependency is "neutralized" by capturing the
/// coefficients and intercept values. Users of the api must maintain the names of
/// the factors.
///
/// Lifetime is tied to objective
///
pub struct Findings<'a> {
    pub all_betas: DVector<f64>,
    pub coefficients: DVector<f64>,
    pub intercept: f64,
    pub objective: &'a Objective,
}
impl<'a> Findings<'a> {
    pub fn report(&self) -> Result<String> {
        let report = format!(
            r#"
-----------------------------------
features: {}
records: {}
coefficients: {}
intercept: {}
AUC score: {}
-----------------------------------
"#,
            self.objective.feature_count(),
            self.objective.x.shape().0,
            self.coefficients,
            self.intercept,
            auc_score(&self.objective.y, &self.predict(true))?,
        );

        Ok(report)
    }
    /*
    pub fn coefficients(&self) -> &DVector<f64> {
        &self.coefficients
    }
    pub fn intercept(&self) -> f64 {
        self.intercept
    } */
    /// Standalone prediction that takes objective and findings
    pub fn predict(&self, binary: bool) -> Prediction<f64> {
        let x = &self.objective.x;
        let coeff: &DVector<f64> = &self.all_betas;

        event!(Level::DEBUG, "🦀 -----------------------------------");
        event!(Level::DEBUG, "🦀 coeff len: {}", &coeff.len());
        event!(Level::DEBUG, "🦀 row width: {}", &x.shape().1);
        event!(Level::DEBUG, "🦀 -----------------------------------");

        let mut y_hat: DVector<f64> = x * coeff;
        if binary {
            y_hat.apply(|v| {
                if sigmoid(*v) > 0.5 {
                    *v = 1.0;
                } else {
                    *v = 0.0;
                }
            });
        } else {
            y_hat.apply(|v| {
                *v = f64::exp(*v) / (f64::exp(*v) + 1.0);
            });
        }

        Prediction::new(y_hat)
    }
}

pub struct Prediction<T> {
    inner: DVector<T>,
}
impl<T> std::ops::Deref for Prediction<T> {
    type Target = DVector<T>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T> From<Vec<T>> for Prediction<T>
where
    T: Scalar,
{
    fn from(vec: Vec<T>) -> Self {
        Prediction {
            inner: DVector::from_vec(vec),
        }
    }
}
impl<T> From<Prediction<T>> for Vec<T>
where
    T: Scalar,
{
    fn from(prediction: Prediction<T>) -> Self {
        prediction.inner.as_slice().to_vec()
    }
}
impl<T> Prediction<T>
where
    T: std::fmt::Display + std::fmt::Debug,
{
    fn new(vec: DVector<T>) -> Self {
        Prediction { inner: vec }
    }
    pub fn show(&self, sample: usize) {
        let sample = self.inner.iter().take(sample).collect::<Vec<&T>>();

        let message = format!(
            r#"
-----------------------------------
predictions: {:?}
-----------------------------------
"#,
            &sample,
        );
        event!(Level::INFO, "{}", message);
    }
}

// flatten the findig to bewteen 1, -1
pub fn sigmoid<T>(v: T) -> f64
where
    T: Borrow<f64> + Neg<Output = f64> + PartialOrd<f64>,
{
    if v < -40. {
        0.
    } else if v > 40. {
        1.
    } else {
        1. / (1. + f64::exp(-v))
    }
}
