// src/lib.rs
//! [SynCode](https://arxiv.org/abs/2403.01632) is a library for constrained
//! generation. It forces an LLM to generate sequences that satisfy a given LR
//! grammar.
pub mod dfa;
pub mod grammar;
pub mod lexer;
pub mod mask;
pub mod parser;
pub mod table;
pub mod types;
// mod util;

// use pyo3::prelude::*;
// use python_bindings::{PyLexerToken, RustLexer, RustParser};

// /// Given a partial output, compute which tokens can be appended to generate a
// /// grammatically correct continuation.
// pub fn grammar_mask_py(partial_output: &[u8]) -> Vec<bool> {
//     let (accept_sequences, remainder) = parser.parse(partial_output);
//     masker.grammar_mask(accept_sequences, remainder)
// }

// /// A Python module implemented in Rust.
// #[pymodule]
// fn rust_parser(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<RustLexer>()?;
//     m.add_class::<RustParser>()?;
//     m.add_class::<PyLexerToken>()?;
//     Ok(())
// }
