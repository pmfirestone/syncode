// src/types.rs
//! Core types used throughout SynCode.

use regex_automata::dfa::dense;

use std::cmp::PartialEq;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

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
#[derive(Clone)]
pub struct Terminal<'a> {
    /// The name of this terminal in the grammar.
    pub name: &'a str,
    /// The regex describing this terminal.
    pub pattern: &'a str,
    /// The DFA that matches this terminal.
    pub dfa: dense::DFA<Vec<u32>>,
    /// This terminal's priority in lexing.
    pub priority: i32,
}

/// An enumeration for symbols of the grammar, to act as a union type of terminals and nonterminals.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Symbol {
    Terminal(Terminal<'static>),
    NonTerminal(NonTerminal<'static>),
}

/// A single production of the grammar.
///
/// That this is exactly a single production, so productions that can go to
/// more than one outcome have to be represented by more than one production in
/// the grammar.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Production {
    /// The left hand side of the production.
    pub source: NonTerminal<'static>,
    /// The right hand side of the production.
    pub result: Vec<Symbol>,
}

/// A context-free grammar.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Grammar {
    /// The set of symbols that are active in this grammar.
    pub symbol_set: Vec<Symbol>,
    /// The first production; this one is the augmented one added to the grammar.
    pub start_production: Production,
    /// The productions that make up this grammar, including the start_production.
    pub productions: Vec<Production>,
}

/// An item of the item set for LR parsing.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Item {
    /// The production that this item contains.
    pub production: Production,
    /// The position of the dot in the result. Invariant: must be in [0, result.len()].
    pub dot: usize,
    /// The look ahead terminal.
    pub lookahead: Terminal<'static>,
}

/// Action enum for LR parsing.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Consume a terminal from input, going to the indicated state.
    Shift(usize),
    /// Reduce the symbols on the stack according to the production.
    Reduce(Production),
    /// Accept the input.
    Accept,
    /// Fail to accept the input.
    Error,
}

/// An action table is a map from a (state_id, terminal) pair to an action.
pub type ActionTable = HashMap<(usize, Terminal<'static>), Action>;

/// A goto table is a map from a (state_id, nonterminal) pair to a state_id.
pub type GotoTable = HashMap<(usize, NonTerminal<'static>), usize>;

// Implementations.
impl<'a> Terminal<'a> {
    pub fn new(name: &'a str, pattern: &'a str, priority: i32) -> Self {
        let Ok(dfa) = dense::DFA::new(pattern) else {
            panic!(
                "While constructing the terminal {name}, could not build a DFA from the pattern {pattern}"
            )
        };
        Terminal {
            name,
            pattern,
            dfa,
            priority,
        }
    }
}

impl fmt::Display for Terminal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Terminal({}, {}, {})",
            self.name, self.pattern, self.priority
        )
    }
}

impl<'a> fmt::Debug for Terminal<'a> {
    /// We don't care about the DFA for the purpose of printing.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Terminal")
            .field("name", &self.name)
            .field("pattern", &self.pattern)
            .field("priority", &self.priority)
            .finish()
    }
}

impl<'a> Hash for Terminal<'a> {
    /// We don't care about the DFA for the purpose of hashing.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.pattern.hash(state);
        self.priority.hash(state);
    }
}

impl<'a> PartialEq for Terminal<'a> {
    /// We don't care about the DFA for the purpose of equality comparison.
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.pattern == other.pattern && self.priority == other.priority
    }
}

impl<'a> Eq for Terminal<'a> {}

/// A type alias for nonterminals of the grammar, purely for readability.
pub type NonTerminal<'a> = &'a str;

// impl Terminal<'static> {
//     /// Consume a string starting from a state and return the state reached.
//     pub fn advance(&self, state: StateID, input: &[u8]) -> StateID {
//         for &b in input {
//             state = self.dfa.next_state(state, b);
//         }
//         state
//     }
// }
