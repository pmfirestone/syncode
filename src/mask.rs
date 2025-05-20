// src/mask.rd
//! The mask store for SynCode, implementing the basic algorithm from the paper.

use crate::dfa::all_dfa_states;
use crate::parser::Parser;
use crate::types::*;
use core::iter::Iterator;
use regex_automata::dfa::dense;
use regex_automata::{Anchored, dfa::Automaton, util::primitives::StateID, util::start::Config};
use std::collections::HashSet;
use std::iter::zip;
use std::{collections::HashMap, vec::Vec};

/// A DFA mask store stores, for each possible position we could be at in
/// matching each terminal, and for each possible accept sequence that could
/// follow, the binary mask over the vocabulary that indicates the possible
/// acceptable continuations.
///
/// We represent the key in the implementation by a `[crate::types::Terminal]`
/// (which is a proxy for a DFA), a
/// `[regex_automata::util::primitives::StateId]`, which indicates where we are
/// in the accepting of that terminal, and a vector of `Terminal`s representing
/// the accept sequence in question.
///
/// > For an integer α, the DFA mask store Mα is a function defined as Mα :
/// > QΩ × Γα → {0, 1}|V |, where QΩ = ⋃ τ ∈Γ Qτ represents the set of all DFA
/// > states and Γα is a set of α-length terminal sequences. Then Mα(q, Λ) = m is
/// > a binary mask such that t ∈ set(m) if dmatch(t, q, Λ).
type DFAMaskStore = HashMap<(Terminal, StateID, Vec<Terminal>), Vec<bool>>;

/// Compute whether the string could match a sequence of terminals starting at a certain state in the first DFA.
///
/// Given a DFA D(Q, Σ, δ, q0, F ), a string w ∈ Σ∗, a DFA state q ∈ Q and any sequence of terminals Λ = {τf +1, τf +2 . . . τf +d}, dmatch(w, q, Λ) = true, if either of the following conditions hold:
/// 1. δ∗(w, q) ∈ live(Q) or
/// 2. ∃w1 ∈ Σ∗, w2 ∈ Σ+ such that w1.w2 = w, δ∗(w1, q) ∈ F and Λ = {} or
/// 3. ∃w1 ∈ Σ∗, w2 ∈ Σ∗ such that w1.w2 = w, δ∗(w1, q) ∈ F, and dmatch(w2, qτf +10 , {τf +2 . . . τf +d}) = true where qτf +10 is the start state corresponding to the DFA for τf +1.
pub fn dmatch(
    input: &[u8],
    dfa: &dense::DFA<Vec<u32>>,
    starting_state: &StateID,
    accept_sequence: &[Terminal],
) -> bool {
    // We'll need to access the starting state later, but we also need
    // something we can mutate as we feed things into the DFA.
    let mut state: StateID = *starting_state;

    // Case 1: the DFA, starting at this state, consumes the entire input and is still alive.
    for &b in input {
        state = dfa.next_state(state, b);
    }
    // Neither dead nor quit means we could match in the future and so are live.
    if !(dfa.is_dead_state(state) || dfa.is_quit_state(state)) {
        return true;
    }

    // Case 2: The DFA consumes a prefix of the string, leaves a non-zero
    // suffix, and there is no sequence of terminals to follow. Assume that
    // grammars respect the maximum munch principle, so w1 is the maximal
    // matching prefix.
    state = *starting_state; // Reset to initial state.
    let mut index_reached: usize = 0;
    for (i, &b) in input.iter().enumerate() {
        state = dfa.next_state(state, b);
        if dfa.is_dead_state(state) | dfa.is_quit_state(state) {
            // We've failed to match, so stop feeding tokens in.
            break;
        }

        if dfa.is_match_state(state) {
            // We haven't yet failed to match, so keep count of how far
            // we've gotten in the input.
            index_reached = i;
        }
    }

    if index_reached > 0 && accept_sequence.is_empty() {
        // We matched at least some bytes and have no more terminals
        // to check.
        return true;
    }

    // Case 3: A prefix of the string is successfully consumed by the DFA, and
    // dmatch is true starting at the next member of sequence_of_terminals.
    state = *starting_state;
    for (i, &b) in input.iter().enumerate() {
        state = dfa.next_state(state, b);

        if !dfa.is_dead_state(state) {
            // Keep munching as long as we're alive.
            continue;
        }

        if dfa.is_dead_state(state) && i == 0 {
            // We failed on the first character, so give up.
            break;
        }

        // We've consumed at least one byte, the DFA is now dead, and there
        // are more terminals to check.
        if i > 0 && dfa.is_dead_state(state) && !accept_sequence.is_empty() {
            let new_dfa = &accept_sequence[0].dfa;

            return dmatch(
                &input[i - 1..],
                &new_dfa.clone(),
                &new_dfa
                    .start_state(&Config::new().anchored(Anchored::Yes))
                    .unwrap(),
                &accept_sequence[1..],
            );
        }
    }

    // None of the previous cases succeeded, so dmatch is false.
    false
}

/// Compute the mask for a given DFA state, accept sequence, and vocabulary.
///
/// Mα(q, Λ) = m is a binary mask such that t ∈ set(m) if dmatch(t, q, Λ),
/// where t is a string (token in the LLM's vocabulary), q is a DFA state, and
/// Λ is an accept sequence.
pub fn dfa_mask(
    dfa: &dense::DFA<Vec<u32>>,
    starting_state: &StateID,
    accept_sequence: &Vec<Terminal>,
    vocabulary: &Vec<&[u8]>,
) -> Vec<bool> {
    let mut mask: Vec<bool> = Vec::with_capacity(vocabulary.len());
    for token in vocabulary {
        mask.push(dmatch(token, dfa, starting_state, accept_sequence));
    }
    mask
}

/// Compute the grammar mask store.
///
/// The mask store is constructed offline by enumerating all DFA states QΩ,
/// considering all possible terminals in Γ, and all tokens in V. The DFA mask
/// store depends on the set of terminals Γ and the model’s vocabulary V. As a
/// result, a unique mask store is created for each grammar and tokenizer
/// combination, and to enhance efficiency, we cache and reuse this table for
/// future inferences.
pub fn dfa_mask_store(
    lexical_terminals: &Vec<Terminal>,
    model_vocabulary: Vec<&[u8]>,
    parser: &Parser,
    _length_of_terminal_sequences: usize,
) -> DFAMaskStore {
    let all_states = all_dfa_states(lexical_terminals);
    let mut store: DFAMaskStore = HashMap::new();
    for (terminal, state_id) in &all_states {
        // For now, hard-code the lookahead of two terminals.
        let next_terminals = parser.next_terminal(terminal);
        for next_terminal in &next_terminals {
            let after_next_terminals = parser.next_terminal(next_terminal);
            for after_next_terminal in &after_next_terminals {
                let accept_sequence = vec![next_terminal.clone(), after_next_terminal.clone()];
                let dfa = &terminal.dfa;
                store.insert(
                    (terminal.clone(), *state_id, accept_sequence.clone()),
                    dfa_mask(dfa, state_id, &accept_sequence, &model_vocabulary),
                );
            }
        }
    }
    store
}

/// Compute the mask for a given accept sequence and remainder.
///
/// Implement algorithm 2 from the paper.
fn grammar_mask(
    accept_sequences: &HashSet<Vec<Terminal>>,
    remainder: &Token,
    mask_store: DFAMaskStore,
    model_vocabulary: &[&[u8]],
) -> Vec<bool> {
    let mut mask: Vec<bool> = vec![false; model_vocabulary.len()];
    for accept_sequence in accept_sequences {
        let first_terminal = &accept_sequence[0];
        let start_state = first_terminal.start_state();
        let end_state = first_terminal.advance(start_state, &remainder.value);
        // Check whether the DFA ended up in a live state.
        if !(first_terminal.dfa.is_dead_state(end_state)
            || first_terminal.dfa.is_quit_state(end_state))
        {
            // Take the union of the mask we've computed so far and the mask from the store.
            for (i, (cur, new)) in zip(
                mask.clone(),
                // Get the relevant mask out of the store.
                mask_store
                    .get(&(
                        first_terminal.clone(),
                        end_state,
                        accept_sequence[1..].to_vec(),
                    ))
                    .unwrap(),
            )
            .enumerate()
            {
                mask[i] = cur || *new;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmatch_case1() {
        let candidate_string = "abba".as_bytes();
        let terminal = Terminal::new("", r"[ab]*cd", 0);
        let Ok(start_state) = terminal
            .dfa
            .start_state(&Config::new().anchored(Anchored::Yes))
        else {
            panic!();
        };
        assert!(dmatch(candidate_string, &terminal.dfa, &start_state, &[]));
    }

    #[test]
    fn test_dmatch_case2() {
        // False in strict mode, true in overapproximation mode (grammar mask).
        let candidate_string = "abbacdd".as_bytes();
        let terminal = Terminal::new("", r"[ab]*", 0);
        let Ok(start_state) = terminal
            .dfa
            .start_state(&Config::new().anchored(Anchored::Yes))
        else {
            panic!();
        };
        assert!(dmatch(candidate_string, &terminal.dfa, &start_state, &[]));
    }

    #[test]
    fn test_dmatch_case3() {
        // Illustrative example from page 12 of the paper.
        let candidate_string = "_prime():".as_bytes();
        let terminal = Terminal::new("", r"[a-zA-Z_]*", 0);
        let Ok(mut start_state) = terminal
            .dfa
            .start_state(&Config::new().anchored(Anchored::Yes))
        else {
            panic!();
        };
        for &b in "is".as_bytes() {
            start_state = terminal.dfa.next_state(start_state, b);
        }
        let accept_sequence = vec![
            Terminal::new("L_PAREN", r"\(", 0),
            Terminal::new("R_PAREN", r"\)", 0),
        ];
        assert!(dmatch(
            candidate_string,
            &terminal.dfa,
            &start_state,
            &accept_sequence
        ));
    }

    #[test]
    fn dmatch_fails_to_match_with_non_empty_accept_sequence() {
        let candidate_string = "abcab".as_bytes();
        let terminal = Terminal::new("A_OR_B", r"[ab]+", 0);
        let start_state = terminal.start_state();
        let accept_sequence = vec![terminal.clone()];
        assert!(!dmatch(
            candidate_string,
            &terminal.dfa,
            &start_state,
            &accept_sequence
        ));
    }

    // #[test]
    // fn test_dmatch_case3a() {
    //     // Consuming next terminal leaves residual string.
    //     let candidate_string = "abbacde";
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[ab]*");
    //     starting_state.advance("ab");
    //     let accept_sequence = vec![r"c"];
    //     assert!(dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dmatch_ugly_unicode_thing() {
    //     // This is a nasty token from an actual LLM. They've played us for fools.
    //     let mut masker = Masker::new();
    //     let mut starting_state = masker.dfa_builder.build_dfa(r"(?i:0|[1-9]\d*)");
    //     assert!(!dmatch("ĠĠ", &mut starting_state, vec![]));
    // }

    // #[test]
    // fn test_dmatch_supports_unicode_fails() {
    //     // Make sure dmatch works on tokens that are multiple bytes in UTF-8,
    //     // even when the match should fail.
    //     let candidate_string = "³Ġt";
    //     let accept_sequence = vec![];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     assert!(!dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dmatch_supports_unicode_case3() {
    //     // Make sure dmatch works on tokens that are multiple bytes in UTF-8.
    //     let candidate_string = "iÃ³";

    //     let accept_sequence = vec![r"\(", r"\)"];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     assert!(!dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dmatch_fails_case2() {
    //     let candidate_string = "3not an id";
    //     let accept_sequence = vec![];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     assert!(!dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dmatch_accepts_matching_input() {
    //     let candidate_string = "indeed";
    //     let accept_sequence = vec![r"\(", r"\)"];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     assert!(dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dmatch_fails_case3() {
    //     let candidate_string = "3not an id";
    //     let accept_sequence = vec![r"\(", r"\)"];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     assert!(!dmatch(
    //         candidate_string,
    //         &mut starting_state,
    //         accept_sequence
    //     ));
    // }

    // #[test]
    // fn test_dfa_mask_name() {
    //     // Illustrative example from page 13 of the paper.
    //     let vocabulary = vec!["_prime():", ":#", "¡", " hi", "indeed", "n0pe"];
    //     let terminal_sequence = vec![r"\(", r"\)"];
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     starting_state.advance("is");
    //     assert_eq!(
    //         dfa_mask(&mut starting_state, &terminal_sequence, &vocabulary),
    //         vec![true, false, false, false, true, false],
    //     );
    // }

    // #[test]
    // fn test_dfa_mask_store() {
    //     let model_vocabulary = vec!["_prime():", "ĠĠ", "'''", " hi", "indeed", "n0pe"];
    //     let lexical_terminals = vec![r"\(", r"\)", r"[a-zA-Z_]*"];
    //     let store = dfa_mask_store(lexical_terminals, model_vocabulary, 2);
    //     let candidate_string = "is";
    //     let mut starting_state = matcher.dfa_builder.build_dfa(r"[a-zA-Z_]*");
    //     starting_state.advance(candidate_string);
    //     assert_eq!(
    //         store.get(&(starting_state, vec![r"\(", r"\)"])).unwrap(),
    //         &vec![true, false, false, false, true, false],
    //     );
    // }
}
