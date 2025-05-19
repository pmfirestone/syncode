// src/parser.rs
//! The parser for SynCode. Takes in a lexed sequence of tokens and determines
//! the accept sequences that could follow.

use std::collections::HashSet;
use std::fmt;

use crate::lexer::Lexer;
use crate::types::*;

// Rust RFC 1733 introduces this syntax as a way to alias bounds, which would
// make this module much more readable. Unfortunately, as of 2025-05-09, the
// behavior is not yet stable. See
// https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md.

/// The Parser with its tables.
///
/// We do not include the state stack as part of the parser struct, since it is
/// easier by far to handle this struct as an immutable value and keep the
/// stack as an argument that is passed in and out for each call.
#[derive(Clone)]
pub struct Parser<'a> {
    /// This parser's lexer.
    // It's not great to have this be part of the Parser, but the logic of
    // [Parser::parse] requires that the Parser know about the remainder, which
    // is most easily gotten using the [Lexer] directly.
    pub lexer: Lexer<'a>,
    /// The action table. Each entry has an index and maps between a terminal and an action.
    pub action_table: ActionTable,
    /// The goto table. Each entry maps between a state index and a map between
    /// a non-terminal and another state index.
    pub goto_table: GotoTable,
    /// The index of the state to start at.
    pub start_state: usize,
    /// The number of lexical tokens we have parsed so far.
    pub token_index: usize,
}

impl<'a> Parser<'a> {
    /// Return all the terminals that could come after this one, regardless of
    /// the state the parser is in.
    pub fn next_terminal(&'a self, terminal: &Terminal<'static>) -> Vec<Terminal<'static>> {
        let states_that_accept_this_terminal: Vec<usize> = self
            .action_table
            .keys()
            .filter(|key| key.1 == *terminal)
            .map(|key| key.0.clone())
            .collect();
        let mut terminals_that_could_follow_this_one: Vec<Terminal<'static>> = Vec::new();
        for state in states_that_accept_this_terminal {
            terminals_that_could_follow_this_one.extend(self.follow(&vec![state]))
        }
        terminals_that_could_follow_this_one
    }

    /// Return the terminals that the parser will accept in the current state.
    pub fn follow(&'a self, state_stack: &Vec<usize>) -> Vec<Terminal<'static>> {
        self.action_table
            .keys()
            .filter(|key| key.0 == *state_stack.last().unwrap())
            .map(|key| key.1.clone())
            .collect()
    }

    /// Feed a token to the parser and process it according to the LR(1) algorithm.
    ///
    /// The inner loop of the LR parsing algorithm.
    pub fn next(
        &'a self,
        terminal: &Terminal<'a>,
        state_stack: Vec<usize>,
    ) -> Result<Vec<usize>, ParserError<'a>> {
        // This implementation is verbose because of the error handling
        // involved. Perhaps there's a way to make it more streamlined by
        // consolidating the error-managing boiler plate.
        let mut state_stack = state_stack;

        loop {
            // Get the current state.
            let Some(state) = state_stack.last() else {
                return Err(ParserError::StackUnderflow);
            };

            // Get the action for this state and terminal.
            let Some(action) = self.action_table.get(&(*state, terminal.clone())) else {
                return Err(ParserError::InvalidState(*state));
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
                    let size = rule.result.len();

                    if size > 0 {
                        // Pop the appropriate number of states.
                        for _ in 0..size {
                            if state_stack.pop().is_none() {
                                return Err(ParserError::StackUnderflow);
                            }
                        }
                    }

                    // Look up the current state.
                    let Some(current_state) = state_stack.last() else {
                        return Err(ParserError::StackUnderflow);
                    };

                    // Get the next state for this state and nonterminal.
                    let Some(next_state) = self.goto_table.get(&(*current_state, rule.source))
                    else {
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
                        terminal,
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
        &'a self,
        partial_output: &'a [u8],
    ) -> Result<(HashSet<Vec<Terminal<'a>>>, Token<'a>), ParserError<'a>> {
        let mut a0: Vec<Terminal> = Vec::new();
        let mut a1: Vec<Terminal> = Vec::new();

        let Ok((tokens, remainder)) = self.lexer.lex(partial_output) else {
            return Err(ParserError::LexerError);
        }; // FIXME: return an actually useful error.

        let last_token = tokens[tokens.len() - 1].clone();
        let mut state_stack = vec![self.start_state];

        for token in &tokens[..] {
            let Some(terminal) = token.terminal.clone() else {
                return Err(ParserError::InvalidToken);
            };
            // FIXME: There must be a less horrid way to do this.
            let Ok(new_state_stack) = self.next(&terminal, state_stack) else {
                break;
            };
            state_stack = new_state_stack;
            a0 = a1;
            a1 = self.follow(&state_stack);
        }

        // There are two cases for accept sequences. See section 4.5 of the
        // paper and Algorithm 4, lines 15-21.
        let mut accept_sequences: HashSet<Vec<Terminal>> = HashSet::new();
        if last_token == remainder {
            // Case 1: the remainder is the last lexical token.
            let Some(remainder_type) = remainder.clone().terminal else {
                return Err(ParserError::StackUnderflow);
            };
            for terminal in a1 {
                accept_sequences.insert(vec![remainder_type.clone(), terminal]);
            }
            for terminal in a0 {
                accept_sequences.insert(vec![terminal]);
            }
        } else {
            // Case 2: the remainder is some unparsed nonsense.
            for terminal in a1 {
                accept_sequences.insert(vec![terminal]);
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
    LexerError,
    StackUnderflow,
    EmptyStack,
    InvalidState(usize),
    InvalidAction(String),
    /// We got a terminal that doesn't work in this state.
    InvalidTerminal(
        /// Actual.
        Terminal<'a>,
        /// Expected.
        Vec<Terminal<'a>>,
    ),
    InvalidToken,
    SyntaxError(String),
    ConfigError(String),
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    // Terminal definitions to be used throughout tests. Commented out ones may
    // come in handy in future tests but are commented to avoid dead code warnings.
    fn word() -> Terminal<'static> {
        Terminal::new("WORD", r"[a-zA-Z_]\w*", 2)
    }

    // fn string() -> Terminal<'static> {
    //     Terminal::new("STRING", r#"("""[^"]*"""|'''[^']*''')"#, 2)
    // }

    fn space() -> Terminal<'static> {
        Terminal::new("SPACE", "\\s+", 0)
    }

    // fn equals() -> Terminal<'static> {
    //     Terminal::new("EQUALS", "=", 1)
    // }

    // fn dot() -> Terminal<'static> {
    //     Terminal::new("DOT", r"\.", 1)
    // }

    fn dec_number() -> Terminal<'static> {
        Terminal::new("DEC_NUMBER", r"0|[1-9]\d*", 1)
    }

    // fn oct_number() -> Terminal<'static> {
    //     Terminal::new("OCT_NUMBER", r"(?i)0o[0-7]+", 1)
    // }

    // fn bin_number() -> Terminal<'static> {
    //     Terminal::new("BIN_NUMBER", r"(?i)0b[0-1]+", 1)
    // }

    // fn hex_number() -> Terminal<'static> {
    //     Terminal::new("HEX_NUMBER", r"(?i)0x[\da-f]+", 1)
    // }

    // fn float_number() -> Terminal<'static> {
    //     Terminal::new(
    //         "FLOAT_NUMBER",
    //         r"((\d+\.\d*|\.\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))",
    //         1,
    //     )
    // }

    // fn semicolon() -> Terminal<'static> {
    //     Terminal::new("SEMICOLON", ";", 0)
    // }

    // fn newline() -> Terminal<'static> {
    //     Terminal::new("NEWLINE", r"\n", 1)
    // }

    fn star() -> Terminal<'static> {
        Terminal::new("STAR", r"\*", 1)
    }

    fn plus() -> Terminal<'static> {
        Terminal::new("PLUS", r"\+", 1)
    }

    /// A convenience terminal representing the end of the input.
    fn eof() -> Terminal<'static> {
        Terminal::new("$", "", 0)
    }

    // Mega-simple grammar courtesy of https://en.wikipedia.org/wiki/LR_parser.
    fn calc_rules() -> Vec<Production> {
        vec![
            Production {
                source: "goal",
                result: vec![Symbol::NonTerminal("sums"), Symbol::Terminal(eof())],
            },
            Production {
                source: "sums",
                result: vec![
                    Symbol::NonTerminal("sums"),
                    Symbol::Terminal(plus()),
                    Symbol::NonTerminal("products"),
                ],
            },
            Production {
                source: "sums",
                result: vec![Symbol::NonTerminal("products")],
            },
            Production {
                source: "products",
                result: vec![
                    Symbol::NonTerminal("products"),
                    Symbol::Terminal(star()),
                    Symbol::NonTerminal("value"),
                ],
            },
            Production {
                source: "products",
                result: vec![Symbol::NonTerminal("value")],
            },
            Production {
                source: "value",
                result: vec![Symbol::Terminal(dec_number())],
            },
            Production {
                source: "value",
                result: vec![Symbol::Terminal(word())],
            },
        ]
    }

    fn calc_action_table() -> ActionTable {
        let rules = calc_rules();
        HashMap::from([
            ((0, dec_number()), Action::Shift(8)),
            ((0, word()), Action::Shift(9)),
            ((1, plus()), Action::Shift(2)),
            ((1, eof()), Action::Accept),
            ((2, dec_number()), Action::Shift(8)),
            ((2, word()), Action::Shift(9)),
            ((3, star()), Action::Shift(5)),
            ((3, plus()), Action::Reduce(rules[1].clone())),
            ((4, star()), Action::Shift(5)),
            ((4, plus()), Action::Reduce(rules[2].clone())),
            ((4, eof()), Action::Reduce(rules[2].clone())),
            ((5, dec_number()), Action::Shift(8)),
            ((5, word()), Action::Shift(9)),
            ((6, star()), Action::Reduce(rules[3].clone())),
            ((6, plus()), Action::Reduce(rules[3].clone())),
            ((6, eof()), Action::Reduce(rules[3].clone())),
            ((7, star()), Action::Reduce(rules[4].clone())),
            ((7, plus()), Action::Reduce(rules[4].clone())),
            ((7, eof()), Action::Reduce(rules[4].clone())),
            ((8, star()), Action::Reduce(rules[5].clone())),
            ((8, plus()), Action::Reduce(rules[5].clone())),
            ((8, eof()), Action::Reduce(rules[5].clone())),
            ((9, star()), Action::Reduce(rules[6].clone())),
            ((9, plus()), Action::Reduce(rules[6].clone())),
            ((9, eof()), Action::Reduce(rules[6].clone())),
        ])
    }

    fn calc_goto_table() -> GotoTable {
        HashMap::from([
            ((0, "sums"), 1),
            ((0, "products"), 4),
            ((0, "value"), 7),
            ((2, "products"), 3),
            ((2, "value"), 7),
            ((5, "value"), 6),
        ])
    }

    fn calc_parser() -> Parser<'static> {
        let action_table = calc_action_table();
        let goto_table = calc_goto_table();

        let Ok(lexer) = Lexer::new(
            vec![word(), star(), dec_number(), plus(), space()],
            HashSet::from([space()]),
        ) else {
            panic!()
        };

        Parser {
            lexer,
            action_table,
            goto_table,
            start_state: 0,
            token_index: 0,
        }
    }

    #[test]
    fn calc_grammar_step_through_states() {
        let parser = calc_parser();
        let terminals_to_parse: Vec<Terminal> =
            vec![word(), star(), dec_number(), plus(), dec_number()];
        let state_stack: Vec<usize> = vec![parser.start_state];

        let Ok(state_stack) = parser.next(&terminals_to_parse[0], state_stack) else {
            panic!()
        };
        assert_eq!(9, *state_stack.last().unwrap());

        let Ok(state_stack) = parser.next(&terminals_to_parse[1], state_stack) else {
            panic!()
        };
        assert_eq!(5, *state_stack.last().unwrap());

        let Ok(state_stack) = parser.next(&terminals_to_parse[2], state_stack) else {
            panic!()
        };
        assert_eq!(8, *state_stack.last().unwrap());

        let Ok(state_stack) = parser.next(&terminals_to_parse[3], state_stack) else {
            panic!()
        };
        assert_eq!(2, *state_stack.last().unwrap());

        let Ok(state_stack) = parser.next(&terminals_to_parse[4], state_stack) else {
            panic!()
        };
        assert_eq!(8, *state_stack.last().unwrap());
    }

    #[test]
    fn end_to_end_parse() {
        let parser = calc_parser();

        let input = "A * 2 + 1".as_bytes();

        let Ok((accept_sequences, remainder)) = parser.parse(input) else {
            panic!()
        };
        assert_eq!(
            Token {
                value: "1".as_bytes(),
                terminal: Some(dec_number()),
                start_pos: 8,
                end_pos: 9,
                line: 1,
                end_line: 1,
                column: 9,
                end_column: 10
            },
            remainder
        );
        // It's not clear to me exactly what the semantics are here
        // w.r.t. whitespace and other ignored terminals. Also, how do we deal
        // with the possibilty that the remainder could change lexical type,
        // even to types that aren't permitted? As it is, the algorithm allows
        // a DEC_NUMBER to change to a WORD, even though that isn't actually
        // possible in the grammar: semantically, what this says is that
        // instead of being a DEC_NUMBER, the last lexical token could have
        // been a WORD, which is technically true. Nevertheless, we already
        // have enough information to know that the last token couldn't
        // possibly become a WORD and could only continue to be a DEC_NUMBER.
        assert_eq!(
            HashSet::from([
                vec![dec_number(), star()],
                vec![dec_number(), plus()],
                vec![dec_number(), eof()],
                vec![word()],
                vec![dec_number()]
            ]),
            accept_sequences
        );
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
                    "Unexpected token '{:?}' (type: {}) at line {}, column {}. Expected one of: {:?}",
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
            ParserError::LexerError => {
                write!(f, "Lexer error",)
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
            ParserError::InvalidTerminal(actual, _expected) => {
                // FIXME: Implement fmt for Vec<Terminal.>
                write!(
                    f,
                    "Unexpected terminal. Got: {}.",
                    actual // , expected
                )
            }
            ParserError::InvalidToken => {
                write!(
                    f,
                    "Received a token without a terminal that wasn't the remainder."
                )
            }
        }
    }
}
