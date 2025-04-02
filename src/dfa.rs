use pyo3::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, start},
    Anchored,
};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

type DFACache = HashMap<Box<[u8]>, Arc<DFA>>;
type DFA = dense::DFA<Vec<u32>>;

/// A DFA along with its state. Generic to facilitate experiementation with
/// different implementations of DFA.
#[pyclass(eq, hash, frozen)]
#[derive(Clone, Debug)]
pub struct DFAState {
    /// The regex representing this dfa.
    pub regex: Box<[u8]>,
    /// The actual DFA implementation from the library.
    pub dfa: Arc<DFA>,
    /// The state of this DFA. Defaults to the starting state of the DFA.
    pub state_id: StateID,
}

/// Construct DFAs with caching. Only one of these should be instantiated in
/// the lifetime of the program.
#[pyclass]
pub struct DFABuilder {
    cache: DFACache,
}

#[pymethods]
impl DFABuilder {
    /// Initialize with an empty cache.
    #[new]
    pub fn new() -> DFABuilder {
        DFABuilder {
            cache: HashMap::new(),
        }
    }

    /// Return a DFAState, either from the cache or building a new one from scratch.
    // FIXME: Remove the clones from this function to accelerate it further.
    pub fn build_dfa(&mut self, regex: &[u8]) -> DFAState {
        match self.cache.get(regex) {
            Some(dfa) => DFAState::new(regex, dfa.clone()),
            None => {
                let regex_str: &str = &std::str::from_utf8(regex).unwrap();
                let new_dfa = Arc::new(DFA::new(regex_str).unwrap());
                self.cache.insert(regex.into(), new_dfa.clone());
                DFAState::new(regex, new_dfa)
            }
        }
    }
}

#[pymethods]
impl DFAState {
    /// For the Python interface, make advance return the whole DFAState rather than the StateID.
    /// TODO: This is probably a better way to do it than returning the StateID.
    #[pyo3(name = "advance")]
    pub fn py_advance(&self, input: &[u8]) -> DFAState {
        let mut dfa = self.clone();
        for b in input {
            dfa.consume(b);
        }
        dfa.clone()
    }

    #[getter(state_id)]
    fn state_id(&self) -> u32 {
        self.state_id.as_u32()
    }

    #[getter(regex)]
    fn regex(&self) -> &[u8] {
        &self.regex
    }
}

/// A dense implementation of the DFAState abstraction.
impl DFAState {
    /// Encapsulate the kluge necessary to set up the DFA correctly for Syncode's use case.
    fn new(regex: &[u8], dfa: Arc<DFA>) -> DFAState {
        // We always want the DFA to match starting from the beginning of the string.
        let config = start::Config::new().anchored(Anchored::Yes);
        let state_id = dfa.start_state(&config).unwrap();
        DFAState {
            regex: regex.into(),
            dfa,
            state_id,
        }
    }

    /// Convenience function to set the state how we want it.
    pub fn advance(&mut self, input: &[u8]) -> StateID {
        for b in input {
            self.consume(b);
        }
        self.state_id
    }

    /// Consume a byte, starting at the current state, setting and
    /// returning the new state.
    pub fn consume(&mut self, b: &u8) -> StateID {
        self.state_id = self.dfa.next_state(self.state_id, *b);
        self.state_id
    }

    /// Return all states of a dfa by breadth-first search. There exists a private
    /// method that returns an iterator over all states. The suggested alternative
    /// is to traverse the graph manually. See
    /// https://github.com/rust-lang/regex/discussions/1223.
    pub fn states(&self) -> Vec<StateID> {
        let mut queue: VecDeque<StateID> = VecDeque::new();
        let mut explored: Vec<StateID> = Vec::new();

        let start = self
            .dfa
            .start_state(&start::Config::new().anchored(Anchored::Yes))
            .unwrap();

        explored.push(start);
        queue.push_back(start);
        while !queue.is_empty() {
            let current_state = queue.pop_front().unwrap();
            // Iterate over whole alphabet.
            for letter in self.dfa.byte_classes().representatives(0..=255) {
                let next = self.dfa.next_state(current_state, letter.as_u8().unwrap());
                if !explored.contains(&next) {
                    explored.push(next);
                    queue.push_back(next);
                }
            }
            // Special end-of-input transition.
            let next = self.dfa.next_eoi_state(current_state);
            if !explored.contains(&next) {
                explored.push(next);
                queue.push_back(next);
            }
        }
        explored
    }
}

impl PartialEq for DFAState {
    //! Avoid comparing the actual DFAs: the state and regex are enough to establish equality.
    fn eq(&self, other: &Self) -> bool {
        (self.regex == other.regex) & (self.state_id == other.state_id)
    }
}

impl Eq for DFAState {}

impl Hash for DFAState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.regex.hash(state);
        self.state_id.hash(state);
    }
}
/// Compute the union of all states of a list of regexes.
pub fn all_dfa_states(terminals: &Vec<&[u8]>) -> Vec<DFAState> {
    let mut res = Vec::new();
    let mut builder = DFABuilder::new();
    for terminal in terminals.iter() {
        let dfa = builder.build_dfa(terminal);
        for state in dfa.states() {
            res.push(DFAState {
                regex: (*terminal).into(),
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
    fn test_consume_character_match() {
        let mut dfa_state = DFABuilder::new().build_dfa(b"a");
        let mut state = dfa_state.consume(&b"a"[0]);
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(dfa_state.dfa.is_match_state(state));
    }

    #[test]
    fn test_consume_character_fails_to_match() {
        let mut dfa_state = DFABuilder::new().build_dfa(b"a");
        let mut state = dfa_state.consume(&b"b"[0]);
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(!dfa_state.dfa.is_match_state(state));
    }

    #[test]
    fn test_advance_match() {
        let mut dfa_state = DFABuilder::new().build_dfa("[ab¥]*".as_bytes());
        let mut state = dfa_state.advance("aabb¥aab".as_bytes());
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(dfa_state.dfa.is_match_state(state));
    }

    #[test]
    fn test_advance_fails_to_match() {
        let mut dfa_state = DFABuilder::new().build_dfa("[ab]*".as_bytes());
        let mut state = dfa_state.advance("aabba¥ab".as_bytes());
        state = dfa_state.dfa.next_eoi_state(state);
        assert!(!dfa_state.dfa.is_match_state(state));
    }

    #[test]
    fn test_advance() {
        let mut dfa_state = DFABuilder::new().build_dfa(r"[a-zA-Z_]*".as_bytes());
        let state = dfa_state.advance("indeed".as_bytes());
        assert!(dfa_state.dfa.is_match_state(state));
    }
}
