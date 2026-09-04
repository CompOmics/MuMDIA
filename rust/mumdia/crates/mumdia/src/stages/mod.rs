//! Pipeline stages. Each is an independent subcommand reading path-addressable
//! inputs and writing declared Parquet + a report (docs/03_io_layer.md).

pub mod align;
pub mod audit;
pub mod compete;
pub mod convert;
pub mod digest;
pub mod extract;
pub mod features;
pub mod peptidoforms;
pub mod predict_frag;
pub mod prescan;
pub mod quant;
pub mod report;
pub mod rescore;
pub mod rt_im_train;
pub mod run;
pub mod run_experiment;
pub mod search_seed;
