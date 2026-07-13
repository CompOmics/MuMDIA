//! Pipeline stages. Each is an independent subcommand reading path-addressable
//! inputs and writing declared Parquet + a report (PLAN.md Section 3.5).

pub mod align;
pub mod compete;
pub mod convert;
pub mod digest;
pub mod extract;
pub mod features;
pub mod peptidoforms;
pub mod predict_frag;
pub mod quant;
pub mod rescore;
pub mod rt_im_train;
pub mod run;
pub mod search_seed;
