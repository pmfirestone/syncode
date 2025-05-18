// src/dfa.rs
//! The DFA logic for SynCode. Used by `mask` to generate the mask store.

use regex_automata::{
    Anchored,
    dfa::{Automaton, dense},
    util::{primitives::StateID, start},
};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::types::{DFA, Terminal, Token};

type DFACache<'a> = Vec<Rc<DFA>>;

/// Return all states of a dfa by breadth-first search. There exists a private
/// method that returns an iterator over all states. The suggested alternative
/// is to traverse the graph manually. See
/// <https://github.com/rust-lang/regex/discussions/1223>.
pub fn states(dfa: &DFA) -> Vec<StateID> {
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
pub fn all_dfa_states<'a>(terminals: &Vec<Terminal<'a>>) -> Vec<DFAState<'a>> {
    let mut res = Vec::new();
    let mut builder = DFABuilder::new();
    for terminal in terminals.iter() {
        let Ok(dfa) = builder.build_dfa(terminal) else {
            panic!()
        };
        for state in dfa.states() {
            res.push(DFAState {
                terminal: terminal.clone(),
                dfa: dfa.dfa.clone(),
                state_id: state,
            });
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_advance_match() {
        let mut builder = DFABuilder::new();
        let Ok(mut dfa_state) = builder.build_dfa(&Terminal {
            name: "",
            pattern: "r[ab¥]*",
            priority: 0,
        }) else {
            panic!()
        };
        let mut state = dfa_state.advance("aabb¥aab".as_bytes());
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(dfa_state.dfa.is_match_state(state));
    }

    #[test]
    fn test_advance_fails_to_match() {
        let mut builder = DFABuilder::new();
        let Ok(mut dfa_state) = builder.build_dfa(&Terminal {
            name: "",
            pattern: r"[ab]*",
            priority: 0,
        }) else {
            panic!()
        };
        let mut state = dfa_state.advance("aabba¥ab".as_bytes());
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(!dfa_state.dfa.is_match_state(state));
    }
}
