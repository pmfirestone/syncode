// src/dfa.rs
//! DFA helper functions. This module primarily exists to export
//! `all_dfa_states`, which is used to construct the mask store.

use crate::types::Terminal;
use regex_automata::{
    Anchored,
    dfa::{Automaton, dense},
    util::{primitives::StateID, start},
};

use std::collections::VecDeque;

/// Return all states of a dfa by breadth-first search. There exists a private
/// method that returns an iterator over all states. The suggested alternative
/// is to traverse the graph manually. See
/// <https://github.com/rust-lang/regex/discussions/1223>.
fn states(dfa: &dense::DFA<Vec<u32>>) -> Vec<StateID> {
    let mut queue: VecDeque<StateID> = VecDeque::new();
    let mut explored: Vec<StateID> = Vec::new();

    let start = dfa
        .start_state(&start::Config::new().anchored(Anchored::Yes))
        .unwrap();

    explored.push(start);
    queue.push_back(start);
    while !queue.is_empty() {
        let current_state = queue.pop_front().unwrap();
        // Iterate over whole alphabet.
        for letter in dfa.byte_classes().representatives(0..=255) {
            let next = dfa.next_state(current_state, letter.as_u8().unwrap());
            if !explored.contains(&next) {
                explored.push(next);
                queue.push_back(next);
            }
        }
        // Special end-of-input transition.
        let next = dfa.next_eoi_state(current_state);
        if !explored.contains(&next) {
            explored.push(next);
            queue.push_back(next);
        }
    }
    explored
}

/// Compute the union of all states of a list of terminals.
pub fn all_dfa_states(terminals: &Vec<Terminal>) -> Vec<(Terminal, StateID)> {
    let mut res = Vec::new();
    for terminal in terminals.iter() {
        let dfa = &terminal.dfa;
        for state in states(dfa) {
            res.push((terminal.clone(), state));
        }
    }
    res
}
