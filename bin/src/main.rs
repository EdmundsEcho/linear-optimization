use color_eyre::eyre::Result;
use tracing::{event, Level};
use tracing_subscriber;

use std::time::Instant;

use propensity_score::prelude::*;

const FILENAME: &str = "/Users/edmund/Downloads/creditcard.v2.csv";

/// Called by some internal process that knows to put the target
/// data in the first column. The data is a "dense matrix".  A single array of floats.
fn main() -> Result<()> {
    // performance and debugging metrics
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let start = Instant::now();

    // build the objective
    // memory is allocated when from file
    let objective = Objective::from_csv(FILENAME, false)?;

    let cfg = CfgBuilder::new().max_iters(100).logging(false).build();

    let findings = logit::run(&objective, cfg)?;

    let duration = start.elapsed();

    findings.report()?;

    let y_hat = findings.predict(false); // binary = false, show_sample
    y_hat.show(5);

    event!(
        Level::INFO,
        "Time elapsed in expensive_function() is: {:?}",
        duration
    );

    Ok(())
}
