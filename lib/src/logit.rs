use argmin::core::observers::slog_logger::SlogLogger;
use argmin::core::observers::ObserverMode;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::quasinewton::LBFGS;

use color_eyre::eyre::{eyre, Result};
use nalgebra::base::DVector;
use tracing::{event, Level};

use std::iter::zip;

use crate::configurations::*;
use crate::models::{sigmoid, Findings, Objective};

// ✅ Replicates the original
/// use trait to specify how use data to compute objective
impl<'a> CostFunction for &'a Objective {
    type Param = DVector<f64>;
    type Output = f64;

    // the loss/cost function
    #[tracing::instrument]
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let ws = param;

        /*
        event!(Level::DEBUG, "🦀 -----------------------------------");
        event!(Level::DEBUG, "🦀 param width: {}", &ws.len());
        event!(Level::DEBUG, "🦀 row width: {}", self.x.row(0).len());
        event!(Level::DEBUG, "🦀 -----------------------------------");
        */

        assert!(
            self.x.row(0).len() == ws.len(),
            "🦀 x feature count not matching guess param len"
        );
        assert!(
            self.feature_count() == ws.len(),
            "🦀 feature count not matching guess size"
        );

        // the guess includes a slot for the intercept/bias
        // create a view that clips the first value
        let cost: f64 = (&self.x * ws)
            .iter_mut()
            .map(|&mut raw_y_hat| sigmoid(raw_y_hat))
            .zip(&self.y)
            .map(|(y_hat, yi)| yi * y_hat.ln() + (1.0 - yi) * (1.0 - y_hat).ln())
            .sum();

        Ok(-cost)
    }
}

// ✅ Replicates the original
/// First or second derivative to help find max and min
impl<'a> Gradient for &'a Objective {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    #[tracing::instrument]
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let ws = param;
        let n = self.feature_count();

        let dyi_x_n: DVector<f64> = (&self.x * ws).map(|raw_y_hat| sigmoid(raw_y_hat)) - &self.y;

        // zip fold
        // Note: For now unable to accomplish this without using Vec instead of DVector
        let result: Vec<f64> =
            zip(self.x.row_iter(), &dyi_x_n).fold(vec![0.0; n], |acc, (xs, dyi)| {
                // event!(Level::DEBUG, "\n🦀 acc: {:?}", &acc);
                // zip map
                zip(acc, xs.iter())
                    .map(|(mut acc_j, x_j)| {
                        acc_j += x_j * dyi;
                        acc_j
                    })
                    .collect()
            });

        Ok(result.into())
    }
}

// #[tracing::instrument]
pub fn run<'a>(
    objective: &'a Objective,
    Cfg {
        max_iters, logging, ..
    }: Cfg,
) -> Result<Findings> {
    // Enter the span, returning a guard object.

    event!(
        Level::INFO,
        "🟢 Running the optimization with feature count: {}",
        objective.feature_count()
    );

    let p = objective.feature_count();

    // Define initial parameter vector
    let init_param: DVector<f64> = DVector::from_vec(vec![0f64; p]);

    // Set condition
    let cond = ArmijoCondition::new(0.5).map_err(|e| eyre!("Failed condition {}", e))?;

    // set up a line search
    let linesearch = BacktrackingLineSearch::new(cond)
        .rho(0.9)
        .map_err(|e| eyre!("Failed linesearch {}", e))?;

    // Set up solver
    let solver = LBFGS::new(linesearch, 7);

    // Run solver
    let res = Executor::new(objective, solver)
        .configure(|state| state.param(init_param).max_iters(max_iters));
    let res = if logging {
        res.add_observer(SlogLogger::term(), ObserverMode::Always)
    } else {
        res
    };
    let res = res.run().map_err(|e| eyre!("Result failed: {}", e))?;

    let w: &DVector<f64> = &res.state().best_param.as_ref().unwrap();

    // std::thread::sleep(std::time::Duration::from_secs(1));

    event!(Level::INFO, "🏁 shape: {:?}", w.shape());

    Ok(Findings {
        all_betas: w.rows(0, p).into_owned(),
        coefficients: w.rows(0, p - 1).into_owned(),
        intercept: w[p - 1],
        objective,
    })
}
