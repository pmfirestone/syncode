// src/lib.rs
//! [SynCode](https://arxiv.org/abs/2403.01632) is a library for constrained
//! generation. It forces an LLM to generate sequences that satisfy a given LR
//! grammar.
pub mod dfa;
pub mod lexer;
pub mod mask;
pub mod parser;
// mod python_bindings;
// mod util;

// use pyo3::prelude::*;
// use python_bindings::{PyLexerToken, RustLexer, RustParser};

// /// A Python module implemented in Rust.
// #[pymodule]
// fn rust_parser(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<RustLexer>()?;
//     m.add_class::<RustParser>()?;
//     m.add_class::<PyLexerToken>()?;
//     Ok(())
// }
