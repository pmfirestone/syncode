// src/types.rs
//! Core types used throughout SynCode.

use regex_automata::dfa::dense;
use std::fmt;

/// A convenience alias for the type of DFA we are using.
pub type DFA = dense::DFA<u32>;

/// A lexical token, what the lexer breaks the input into.
#[derive(Clone, Debug, PartialEq)]
pub struct Token<'a> {
    /// The content of the token.
    pub value: &'a [u8],
    /// The type of terminal that this is in the grammar. None if this token
    /// couldn't be lexed, which can happen in the case that this is the
    /// unlexable remainder.
    pub terminal: Option<Terminal<'a>>,
    /// Where in the input the token begins.
    pub start_pos: usize,
    /// Where in the input the token ends.
    pub end_pos: usize,
    /// The line of the input the token begins on.
    pub line: usize,
    /// The line of the input the token ends on.
    pub end_line: usize,
    /// The column of the input the token begins on.
    pub column: usize,
    /// The column of the input the token ends on.
    pub end_column: usize,
}

/// A terminal of the grammar.
///
/// FIXME: As a future optimization, put as many of these as possible behind
/// `Rc`s or `Arc`s, because they are immutable and are often copied or moved
/// around.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Terminal<'a> {
    /// The name of this terminal in the grammar.
    pub name: &'a str,
    /// The regex describing this terminal.
    pub pattern: &'a str,
    /// This terminal's priority in lexing.
    pub priority: i32,
}

/// A type alias for nonterminals of the grammar, purely for readability.
pub type NonTerminal<'a> = &'a str;

// Implementations for the types above.
// impl fmt::Display for Terminal<'_> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "Terminal({}, {}, {})",
//             self.name, self.pattern, self.priority
//         )
//     }
// }

// impl Terminal<'static> {
//     /// Consume a string starting from a state and return the state reached.
//     pub fn advance(&self, state: StateID, input: &[u8]) -> StateID {
//         for &b in input {
//             state = self.dfa.next_state(state, b);
//         }
//         state
//     }
// }

// impl fmt::Display for Token<'_> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "Token({:?}, {})", self.terminal, self.value)
//     }
// }
