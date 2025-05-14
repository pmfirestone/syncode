// src/parser.rs
//! The parser for SynCode. Takes in a lexed sequence of tokens and determines
//! the accept sequences that could follow.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use crate::lexer::{Lexer, NonTerminal, Terminal, Token};

// Rust RFC 1733 introduces this syntax as a way to alias bounds, which would
// make this module much more readable. Unfortunately, as of 2025-05-09, the
// behavior is not yet stable. See
// https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md.

// trait ParserStateIndex = Clone + Eq + Hash + std::fmt::Debug;

/// Rule represents a grammar production rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rule<'a> {
    pub id: usize,
    pub origin: &'a str,
    pub expansion: Vec<&'a str>,
}

impl fmt::Display for Rule<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.origin, self.expansion.join(" "))
    }
}

/// Action enum for LR parsing.
#[derive(Clone, Debug, PartialEq)]
pub enum Action<'a, S: std::fmt::Debug> {
    Shift(S),
    Reduce(Rule<'a>),
    Accept,
    Error,
}

type ActionTable<'a> = HashMap<usize, HashMap<Terminal<'a>, Action<'a, usize>>>;
type GotoTable<'a> = HashMap<usize, HashMap<NonTerminal<'a>, usize>>;

/// The Parser with its states, stack, and table.
#[derive(Clone)]
pub struct Parser<'a> {
    /// The lexer this parser uses.
    lexer: Lexer<'a>,
    /// The action table. Each entry has an index and maps between a terminal and an action.
    pub action_table: ActionTable<'a>,
    /// The goto table. Each entry maps between a state index and a map between
    /// a non-terminal and another state index.
    pub goto_table: GotoTable<'a>,
    pub start: &'a str,
    pub start_state: usize,
    pub end_state: usize,
    /// The number of lexical tokens we have parsed so far.
    pub token_index: usize,
    /// The position in the input we have parsed to so far.
    pub last_pos: usize,
}

impl<'a> Parser<'a> {
    /// Return the terminals that the parser will accept in the current state.
    pub fn follow(&'a self, state_stack: &Vec<usize>) -> Vec<Terminal<'a>> {
        // Get the names of the terminals that can follow in the current state.
        let terminal_names = self
            .action_table
            .get(state_stack.last().unwrap())
            .unwrap();

        // Look up the Terminal structs that represent the currently-acceptable terminals.
        self.lexer
            .terminals
            .iter()
            .filter(|terminal| terminal_names.contains_key(terminal))
            .cloned()
            .collect::<Vec<Terminal>>()
    }

    /// Feed a token to the parser and process it according to the LR(1) algorithm.
    ///
    /// The inner loop of the LR parsing algorithm.
    pub fn next(&'a self, token: Token<'a>, state_stack: Vec<usize>) -> Result<Vec<usize>, ParserError<'a>> {
        // This implementation is verbose because of the error handling
        // involved. Perhaps there's a way to make it more streamlined by
        // consolidating the error-managing boiler plate.
	let mut state_stack = state_stack;
	
        loop {
            // Get the current state.
            let Some(state) = state_stack.last() else {
                return Err(ParserError::StackUnderflow);
            };

            // Get the actions in the current state.
            let Some(actions) = self.action_table.get(&state) else {
                return Err(ParserError::InvalidState(*state));
            };

            // Look up the current state and token type in the parse table.
            // FIXME: This will panic if it gets a token that is the unlexed remainder.
            let Some(action) = actions.get(&token.terminal.clone().unwrap()) else {
                return Err(ParserError::UnexpectedToken {
                    token: token.clone(),
                    expected: self.follow(&state_stack),
                    state_index: self.token_index,
                });
            };

            // eprintln!("Current state: {:?}, Token: {:?}", state, token);
            // eprintln!("Action: {:?}", action);
            // eprintln!("Transitions: {:?}", states.get(&state));

            // Dispatch on action types.
            match action {
                Action::Shift(next_state) => {
                    // Just push next state on shift.
                    state_stack.push(*next_state);
                    return Ok(state_stack); // Not yet accepted.
                }

                Action::Reduce(rule) => {
                    // On a reduce, pop states according to the rule expansion length.
                    let size = rule.expansion.len();

                    if size > 0 {
                        // Pop the appropriate number of states.
                        for _ in 0..size {
                            if state_stack.pop().is_none() {
                                return Err(ParserError::StackUnderflow);
                            }
                        }
                    }

                    // Look up the next state in the goto table.
                    let Some(current_state) = state_stack.last() else {
                        return Err(ParserError::StackUnderflow);
                    };

                    // Get the gotos for this state.
                    let Some(gotos) = self.goto_table.get(&current_state) else {
                        return Err(ParserError::InvalidState(*current_state));
                    };

                    // Look up the next state in the goto table based on the rule.
                    let Some(next_state) = gotos.get(rule.origin) else {
                        return Err(ParserError::InvalidState(*current_state));
                    };

                    // Make this the new current state.
                    state_stack.push(*next_state);
                }

                Action::Accept => {
                    // We're probably never going to reach this case, and there
                    // isn't really anything for us to do if we do.
                    return Ok(state_stack);
                }

                _ => {
                    // Anything else is an Error action.
                    return Err(ParserError::SyntaxError(format!(
                        "Parser error at token: {}",
                        token,
                    )));
                }
            }
        }
    }

    /// Parse tokens without building a tree, just producing the accept
    /// sequences and remainder. This is Algorithm 4 from the paper.
    ///
    /// Take in the partial output the model has generated so far and return
    /// the accept sequences and the unparsed or lexed remainder.
    // Don't implement the cache and restore behavior yet; just reparse
    // from scratch each time. We'll see whether it's a problem in
    // benchmarking and come back for it if we need to.
    pub fn parse(
        &'a mut self,
        partial_output: &'a str,
    ) -> Result<(Vec<Vec<Terminal<'a>>>, Token<'a>), ParserError<'a>> {
        let mut a0: Vec<Terminal> = Vec::new();
        let mut a1: Vec<Terminal> = Vec::new();

        let Ok((tokens, remainder)) = self.lexer.lex(&*partial_output) else {
            return Err(ParserError::EmptyStack);
        }; // FIXME: return an actually useful error.

        let last_token = tokens[tokens.len() - 1].clone();
	let mut state_stack = vec![self.start_state];
	
        for token in &tokens[..] {
	    let Ok(new_state_stack) = self.next(token.clone(), state_stack) else {
                break;
	    };
	    state_stack = new_state_stack;
            a0 = a1;
            a1 = self.follow(&state_stack);
        }

        // There are two cases for accept sequences. See section 4.5 of the
        // paper and Algorithm 4, lines 15-21.
        let mut accept_sequences: Vec<Vec<Terminal>> = Vec::new();
        if last_token == remainder {
            // Case 1: the remainder is the last lexical token.
            let Some(remainder_type) = remainder.clone().terminal else {
                return Err(ParserError::StackUnderflow);
            };
            for terminal in a1 {
                accept_sequences.push(vec![remainder_type.clone(), terminal]);
            }
            for terminal in a0 {
                accept_sequences.push(vec![terminal]);
            }
        } else {
            // Case 2: the remainder is some unparsed nonsense.
            for terminal in a1 {
                accept_sequences.push(vec![terminal]);
            }
        }
        return Ok((accept_sequences, remainder));
    }
}

// Error types for the parser
#[derive(Debug, Clone)]
pub enum ParserError<'a> {
    UnexpectedToken {
        token: Token<'a>,
        expected: Vec<Terminal<'a>>,
        state_index: usize,
    },
    UnexpectedEof,
    LexerError {
        error_type: String,
        pos: usize,
        line: usize,
        column: usize,
        char: char,
    },
    StackUnderflow,
    EmptyStack,
    InvalidState(usize),
    InvalidAction(String),
    SyntaxError(String),
    ConfigError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORD: Terminal = Terminal {
        name: "WORD",
        pattern: "\\w+",
        priority: 2,
    };

    const DEC_NUMBER: Terminal = Terminal {
        name: "DEC_NUMBER",
        pattern: r"0|[1-9]\d*",
        priority: 1,
    };

    const EOF: Terminal = Terminal {
        name: "EOF",
        pattern: "",
        priority: 0,
    };

    const STAR: Terminal = Terminal {
        name: "STAR",
        pattern: r"\*",
        priority: 1,
    };

    const PLUS: Terminal = Terminal {
        name: "PLUS",
        pattern: r"\+",
        priority: 1,
    };

    #[test]
    fn calc_grammar() {
        // Mega-simple grammar courtesy of https://en.wikipedia.org/wiki/LR_parser.
        let rules: Vec<Rule> = vec![
            Rule {
                id: 0,
                origin: "goal",
                expansion: vec!["sums", "EOF"],
            },
            Rule {
                id: 1,
                origin: "sums",
                expansion: vec!["sums", "PLUS", "products"],
            },
            Rule {
                id: 2,
                origin: "sums",
                expansion: vec!["products"],
            },
            Rule {
                id: 3,
                origin: "products",
                expansion: vec!["products", "STAR", "value"],
            },
            Rule {
                id: 4,
                origin: "products",
                expansion: vec!["value"],
            },
            Rule {
                id: 5,
                origin: "value",
                expansion: vec!["DEC_NUMBER"],
            },
            Rule {
                id: 6,
                origin: "value",
                expansion: vec!["WORD"],
            },
        ];

        let action_table: ActionTable = HashMap::from([
            (
                0,
                HashMap::from([(DEC_NUMBER, Action::Shift(8)), (WORD, Action::Shift(9))]),
            ),
            (
                1,
                HashMap::from([(PLUS, Action::Shift(2)), (EOF, Action::Accept)]),
            ),
            (
                2,
                HashMap::from([(DEC_NUMBER, Action::Shift(8)), (WORD, Action::Shift(9))]),
            ),
            (
                3,
                HashMap::from([
                    (STAR, Action::Shift(5)),
                    (PLUS, Action::Reduce(rules[1].clone())),
                ]),
            ),
            (
                4,
                HashMap::from([
                    (STAR, Action::Shift(5)),
                    (PLUS, Action::Reduce(rules[2].clone())),
                    (EOF, Action::Reduce(rules[2].clone())),
                ]),
            ),
            (
                5,
                HashMap::from([(DEC_NUMBER, Action::Shift(8)), (WORD, Action::Shift(9))]),
            ),
            (
                6,
                HashMap::from([
                    (STAR, Action::Reduce(rules[3].clone())),
                    (PLUS, Action::Reduce(rules[3].clone())),
                    (EOF, Action::Reduce(rules[3].clone())),
                ]),
            ),
            (
                7,
                HashMap::from([
                    (STAR, Action::Reduce(rules[4].clone())),
                    (PLUS, Action::Reduce(rules[4].clone())),
                    (EOF, Action::Reduce(rules[4].clone())),
                ]),
            ),
            (
                8,
                HashMap::from([
                    (STAR, Action::Reduce(rules[5].clone())),
                    (PLUS, Action::Reduce(rules[5].clone())),
                    (EOF, Action::Reduce(rules[5].clone())),
                ]),
            ),
            (
                9,
                HashMap::from([
                    (STAR, Action::Reduce(rules[6].clone())),
                    (PLUS, Action::Reduce(rules[6].clone())),
                    (EOF, Action::Reduce(rules[6].clone())),
                ]),
            ),
        ]);

        let goto_table: GotoTable = HashMap::from([
            (
                0,
                HashMap::from([("sums", 1), ("products", 4), ("value", 7)]),
            ),
            (2, HashMap::from([("products", 3), ("value", 7)])),
            (5, HashMap::from([("value", 6)])),
        ]);

        let terminals: Vec<Terminal> = vec![WORD, STAR, DEC_NUMBER, PLUS, DEC_NUMBER];
    }
}

impl fmt::Display for ParserError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::UnexpectedToken {
                token,
                expected,
                state_index: _,
            } => {
                write!(
                    f,
                    "Unexpected token '{}' (type: {}) at line {}, column {}. Expected one of: {:?}",
                    token.value,
                    token.terminal.clone().unwrap().name,
                    token.line,
                    token.column,
                    expected
                )
            }
            ParserError::UnexpectedEof => {
                write!(f, "Unexpected end of input")
            }
            ParserError::LexerError {
                error_type,
                pos,
                line,
                column,
                char,
            } => {
                write!(
                    f,
                    "Lexer error: {} at position {} (line {}, column {}): '{}'",
                    error_type, pos, line, column, char
                )
            }
            ParserError::StackUnderflow => {
                write!(f, "Parser stack underflow")
            }
            ParserError::EmptyStack => {
                write!(f, "Parser stack is empty")
            }
            ParserError::InvalidState(msg) => {
                write!(f, "Invalid parser state: {}", msg)
            }
            ParserError::InvalidAction(msg) => {
                write!(f, "Invalid parser action: {}", msg)
            }
            ParserError::SyntaxError(msg) => {
                write!(f, "Syntax error: {}", msg)
            }
            ParserError::ConfigError(msg) => {
                write!(f, "Parser configuration error: {}", msg)
            }
        }
    }
}
