///
/// Configuration for the optimization process
///
pub struct CfgBuilder {
    max_iters: u64,
    logging: bool,
    cfg_predict: Option<CfgPredict>,
}

impl CfgBuilder {
    pub fn new() -> CfgBuilder {
        CfgBuilder {
            max_iters: 100,
            logging: false,
            cfg_predict: None,
        }
    }

    pub fn max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }

    pub fn logging(mut self, logging: bool) -> Self {
        self.logging = logging;
        self
    }

    pub fn with_predict(mut self, cfg_predict: CfgPredict) -> Self {
        self.cfg_predict = Some(cfg_predict);
        self
    }

    pub fn build(self) -> Cfg {
        Cfg {
            max_iters: self.max_iters,
            logging: self.logging,
            cfg_predict: self.cfg_predict,
        }
    }
}

pub struct Cfg {
    pub max_iters: u64,
    pub logging: bool,
    pub cfg_predict: Option<CfgPredict>,
}

pub struct CfgPredict {
    pub binary_output: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        CfgBuilder::new().build()
    }
}

impl Default for CfgPredict {
    fn default() -> Self {
        CfgPredict {
            binary_output: false,
        }
    }
}
