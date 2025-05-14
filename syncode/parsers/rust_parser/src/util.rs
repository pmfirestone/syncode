// src/util.rs
//! Utilities for SynCode.
use crate::lexer::{Lexer, Terminal};
use crate::parser::{Action, Parser, Rule};
use std::collections::HashMap;

/// Helper function to build a `Parser` out of serialized forms from Lark.
pub fn load_parser<'a>(
    rules: &HashMap<usize, Rule>,
    states_dict: HashMap<String, HashMap<String, (String, String)>>,
    start: &str,
    start_state: usize,
    end_state: usize,
) -> Parser<'a> {
    let mut states: HashMap<usize, HashMap<<'a>, Action<usize>>> = HashMap::new();
    let mut start_states: HashMap<String, usize> = HashMap::new();
    let mut end_states: HashMap<String, usize> = HashMap::new();

    start_states.insert(start.to_string(), start_state);
    end_states.insert(start.to_string(), end_state);

    // Convert serialized states to a ParseTable
    for (state_str, transitions) in states_dict {
        let state = state_str.parse::<usize>().unwrap_or(0);
        let mut state_transitions = HashMap::new();

        for (symbol, (action_type, action_value)) in transitions {
            // eprintln!("Action: '{}' -> '{}'", action_type, action_value);

            let action = match action_type.as_str() {
                "shift" => Action::Shift(action_value.parse::<usize>().unwrap_or(0)),
                "reduce" => {
                    let rule_id = action_value.parse::<usize>().unwrap_or(0);
                    if let Some(rule) = rules.get(&rule_id) {
                        Action::Reduce(rule.clone())
                    } else {
                        // Default to an empty rule if not found
                        Action::Reduce(Rule {
                            id: rule_id,
                            origin: "unknown",
                            expansion: vec![],
                        })
                    }
                }
                // "accept" => Action::Accept,
                _ => Action::Error,
            };

            state_transitions.insert(symbol.clone(), action);
        }

        states.insert(state, state_transitions);
    }

    Parser {
        lexer: Lexer::new(),
        states,
        start_states,
        end_states,
        start: start.to_string(),
        start_state,
        end_state,
        state_stack: vec![],
        token_index: 0,
        last_pos: 0,
    }
}
