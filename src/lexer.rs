// src/lexer.rs
//! The lexer for SynCode. The primary procedure for this module is `lex`
//! (q.v.), which takes a text and returns a sequence of lexical tokens along
//! with a "remainder". See the paper for more detail.
use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::{Anchored, util::start};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::types::{Terminal, Token};

/// A type to describe errors that can arise in lexing.
#[derive(Debug, Clone)]
pub enum LexError {
    UnexpectedChar {
        pos: usize,
        line: usize,
        column: usize,
        allowed: Vec<String>,
        char: char,
    },
    Eof {
        pos: usize,
        line: usize,
        column: usize,
    },
    InitError(String),
    RegexError(String),
}

/// Hold DFAs for the terminals in the grammar.
#[derive(Clone)]
struct Scanner {
    /// The DFA for matching patterns.
    dfa: dense::DFA<Vec<u32>>,
    /// Maps DFA match pattern to the TerminalDef it represents.
    index_to_type: HashMap<usize, Terminal>,
    /// Maps token type name to whether it can contain newlines.
    _newline_types: HashSet<Arc<str>>,
    /// Terminal definitions for reference.
    _terminals: Vec<Terminal>,
    /// All allowed types.
    pub _allowed_types: HashSet<Arc<str>>,
}

impl Scanner {
    pub fn new(terminals: Vec<Terminal>) -> Result<Self, LexError> {
        let mut newline_types = HashSet::new();
        let mut allowed_types = HashSet::with_capacity(terminals.len());
        let mut index_to_type = HashMap::with_capacity(terminals.len());

        // Determine which patterns might contain newlines
        for terminal in &terminals {
            let pattern_str = &terminal.pattern;
            if pattern_str.contains("\\n")
                || pattern_str.contains("\n")
                || pattern_str.contains("\\s")
                || pattern_str.contains("[^")
                || (pattern_str.contains(".") && pattern_str.contains("(?s"))
            {
                newline_types.insert(terminal.name.clone());
            }

            allowed_types.insert(terminal.name.clone());
        }

        // Sort terminals by priority (highest first)
        let mut sorted_terminals = terminals.clone();
        sorted_terminals.sort_by(|a, b| {
            let prio_cmp = b.priority.cmp(&a.priority);
            if prio_cmp != std::cmp::Ordering::Equal {
                return prio_cmp;
            }

            // If priorities are equal, sort by pattern length (descending)
            b.pattern.len().cmp(&a.pattern.len())
        });

        // Create patterns for the DFA
        let mut patterns = Vec::with_capacity(sorted_terminals.len());

        // Process each terminal
        for (i, terminal) in sorted_terminals.iter().enumerate() {
            index_to_type.insert(i, terminal.clone());
            patterns.push(terminal.pattern.clone());
        }

        // Build the DFA
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .minimize(true) // Minimize the DFA for better performance
                    .start_kind(StartKind::Anchored),
            ) // Only match from the start of the input
            .build_many(&patterns)
            .map_err(|e| LexError::RegexError(format!("Failed to build DFA: {}", e)))?;

        Ok(Scanner {
            dfa,
            index_to_type,
            _newline_types: newline_types,
            _terminals: sorted_terminals,
            _allowed_types: allowed_types,
        })
    }

    /// Match the next token in the input, beginning at position pos, and
    /// return it along with the type of terminal that it is.
    ///
    /// Look for the longest possible match.
    pub fn match_token(&self, text: Box<[u8]>, pos: usize) -> Option<(Box<[u8]>, &Terminal)> {
        if pos >= text.len() {
            return None;
        }

        let rest = &text[pos..];

        let config = start::Config::new().anchored(Anchored::Yes);
        let mut state = self.dfa.start_state(&config).expect("no look-around");

        if self.dfa.is_dead_state(state) {
            return None;
        }

        // Keep track of the best match so far
        let mut best_match: Option<(usize, usize)> = None; // (pattern_idx, length)
        let mut current_len = 0;

        // Walk through the DFA state by state
        for &byte in rest {
            state = self.dfa.next_state(state, byte);

            if self.dfa.is_dead_state(state) {
                break;
            }

            let eoi_state = self.dfa.next_eoi_state(state);
            current_len += 1;

            if self.dfa.is_match_state(eoi_state) {
                let pattern_idx = self.dfa.match_pattern(eoi_state, 0).as_usize();

                match best_match {
                    None => {
                        best_match = Some((pattern_idx, current_len));
                    }
                    Some((_, len)) if current_len > len => {
                        // Prefer longer matches
                        best_match = Some((pattern_idx, current_len));
                    }
                    _ => {} // Keep existing best match
                }
            }
        }

        // Return the best match found as string slices
        if let Some((pattern_idx, match_len)) = best_match {
            if let Some(terminal) = self.index_to_type.get(&pattern_idx) {
                return Some((rest[..match_len].into(), terminal));
            }
        }

        None
    }
}

/// A lexer.
#[derive(Clone)]
pub struct Lexer {
    /// The machinery for the DFAs.
    scanner: Scanner,
    /// The terminals this lexer recognizes.
    pub terminals: Vec<Terminal>,
    /// The terminals that this lexer ignores.
    pub ignore_types: HashSet<Terminal>,
    /// The terminals that contain newlines.
    pub newline_types: HashSet<Terminal>,
}

impl Lexer {
    /// Construct a new lexer that recognizes the given `terminals` and ignores
    /// the `ignore_types`.
    ///
    /// Note that all members of `ignore_types` must also be in `terminals`:
    /// otherwise they won't be recognized at all. This is perhaps suboptimal
    /// API design.
    pub fn new(
        terminals: Vec<Terminal>,
        ignore_types: HashSet<Terminal>,
    ) -> Result<Self, LexError> {
        // Determine which patterns might contain newlines
        let mut newline_types: HashSet<Terminal> = HashSet::new();
        for terminal in &terminals {
            if terminal.pattern.contains("\\n")
                || terminal.pattern.contains("\n")
                || terminal.pattern.contains("\\s")
                || terminal.pattern.contains("[^")
                || (terminal.pattern.contains(".") && terminal.pattern.contains("(?s"))
            {
                newline_types.insert(terminal.clone());
            }
        }

        // Create scanner
        match Scanner::new(terminals.clone()) {
            Ok(scanner) => Ok(Lexer {
                scanner,
                terminals,
                ignore_types,
                newline_types,
            }),
            Err(e) => Err(e),
        }
    }

    /// Get the next token from text, updating pos, line, and column to the end
    /// of the new token. Return a flag saying whether or not this token is the remainder.
    // An alternative design would be to distinguish between remainder and
    // non-remainder by adding a member to the Token struct, or by using an
    // entirely different type for it. Since the remainder and lexed sequence
    // are generally handled separately, I think it's best to simply flag to
    // the caller (`lex`, which is the outer loop here) whether or not this
    // procedure is returning the remainder. Then that procedure returns a
    // pair, and its caller in turn unpacks that pair. This keeps the notation
    // in the code similar to that in the paper.
    fn next_token(
        &self,
        text: &[u8],
        mut pos: usize,
        mut line: usize,
        mut column: usize,
    ) -> Result<(Token, bool), LexError> {
        loop {
            // Try to match next token
            if let Some((value, terminal)) = self.scanner.match_token(text.into(), pos) {
                let ignored = self.ignore_types.contains(terminal);

                // If this token is ignored, update position and continue the loop
                if ignored {
                    let contains_newline = self.newline_types.contains(terminal);

                    // Update line and column information
                    if contains_newline {
                        // Calculate new line and column for tokens with newlines
                        for &b in &value {
                            if b == b'\n' {
                                line += 1;
                                column = 1;
                            } else {
                                column += 1;
                            }
                        }
                    } else {
                        column += value.len();
                    }

                    // Move position forward and continue the loop
                    pos += value.len();
                    continue;
                }

                // For non-ignored tokens, create and return the token
                let start_pos = pos;
                let end_pos = start_pos + value.len();
                let start_line = line;
                let start_column = column;

                // Calculate end line and column
                let contains_newline = self.newline_types.contains(terminal);
                let (end_line, end_column) = if contains_newline {
                    // Calculate for tokens with newlines
                    let mut current_line = line;
                    let mut current_column = column;

                    for &b in &value {
                        if b == b'\n' {
                            current_line += 1;
                            current_column = 1;
                        } else {
                            current_column += 1;
                        }
                    }

                    (current_line, current_column)
                } else {
                    // Simple calculation for tokens without newlines
                    (line, column + value.len())
                };

                return Ok((
                    Token {
                        value: value.into(),
                        terminal: Some(terminal.clone()),
                        start_pos,
                        end_pos,
                        line: start_line,
                        column: start_column,
                        end_line,
                        end_column,
                    },
                    false,
                ));
            } else {
                // No match found. Return what's left as the unlexed
                // remainder. The parser will pass this on to the mask store,
                // where, if there's a real error, it will finally be
                // detected. For now, we avoid duplicating the logic necessary
                // to check whether we are dealing with the prefix of a lexical
                // token that may someday become valid or a truly irredeemable
                // error: this will be detected when we attempt partial matches
                // in the mask store.
                let value = &text[pos..];
                return Ok((
                    Token {
                        value: value.into(),
                        terminal: None,
                        start_pos: pos,
                        end_pos: text.len(),
                        line,
                        column,
                        // TODO: How to compute these values?
                        end_line: usize::MAX,
                        end_column: usize::MAX,
                    },
                    true,
                ));
            }
        }
    }

    /// Lex the entire text and return all the tokens along with the remainder.
    ///
    /// The remainder (see sec. 4.2 of the paper) is either the last lexical
    /// token, in the case where the entire input could be lexed, or the
    /// unlexable suffix, in the case where the end of the input could not be
    /// lexed.
    pub fn lex(&self, text: &[u8]) -> Result<(Vec<Token>, Token), LexError> {
        // Pre-allocate a reasonably-sized vector based on estimated token density
        let estimated_token_count = text.len() / 8;
        let mut tokens = Vec::with_capacity(estimated_token_count);

        let mut remainder: Token;

        let mut pos = 0;
        let mut line = 1;
        let mut column = 1;

        // Start timing for performance measurement
        let start_time = std::time::Instant::now();

        loop {
            let (new_token, is_remainder) = self.next_token(text, pos, line, column)?;

            if is_remainder {
                // We should quit early, because we've seen all there is to see.
                let elapsed = start_time.elapsed();
                eprintln!(
                    "Rust lexing completed in {:?} - produced {} tokens",
                    elapsed,
                    tokens.len()
                );
                return Ok((tokens, new_token));
            }

            // Otherwise, continue counting forward to get new tokens.
            pos = new_token.end_pos;
            line = new_token.end_line;
            column = new_token.end_column;

            tokens.push(new_token.clone());

            // The remainder will be the last token we've seen, unless
            // the last thing we see is unlexable.
            remainder = new_token;

            if pos >= text.len() {
                let elapsed = start_time.elapsed();
                eprintln!(
                    "Rust lexing completed in {:?} - produced {} tokens",
                    elapsed,
                    tokens.len()
                );
                return Ok((tokens, remainder));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Terminal definitions to be used throughout tests.
    fn word() -> Terminal {
        Terminal::new("WORD", r"[a-zA-Z_]\w*", 2)
    }

    fn string() -> Terminal {
        Terminal::new("STRING", r#"("""[^"]*"""|'''[^']*''')"#, 2)
    }

    fn space() -> Terminal {
        Terminal::new("SPACE", "\\s+", 0)
    }

    fn equals() -> Terminal {
        Terminal::new("EQUALS", "=", 1)
    }

    fn dot() -> Terminal {
        Terminal::new("DOT", r"\.", 1)
    }

    fn dec_number() -> Terminal {
        Terminal::new("DEC_NUMBER", r"0|[1-9]\d*", 1)
    }

    fn oct_number() -> Terminal {
        Terminal::new("OCT_NUMBER", r"(?i)0o[0-7]+", 1)
    }

    fn bin_number() -> Terminal {
        Terminal::new("BIN_NUMBER", r"(?i)0b[0-1]+", 1)
    }

    fn hex_number() -> Terminal {
        Terminal::new("HEX_NUMBER", r"(?i)0x[\da-f]+", 1)
    }

    fn float_number() -> Terminal {
        Terminal::new(
            "FLOAT_NUMBER",
            r"((\d+\.\d*|\.\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))",
            1,
        )
    }

    fn semicolon() -> Terminal {
        Terminal::new("SEMICOLON", ";", 0)
    }

    fn newline() -> Terminal {
        Terminal::new("NEWLINE", r"\n", 1)
    }

    fn star() -> Terminal {
        Terminal::new("STAR", r"\*", 1)
    }

    fn plus() -> Terminal {
        Terminal::new("PLUS", r"\+", 1)
    }

    #[test]
    fn lexer_initialization() {
        let terminal_defs = vec![word(), space()];

        let ignore_types = HashSet::from([space()]);

        let Ok(lexer) = Lexer::new(terminal_defs, ignore_types) else {
            panic!()
        };

        // Check if it was initialized correctly
        assert_eq!(lexer.terminals.len(), 2);
        assert_eq!(lexer.ignore_types.len(), 1);
    }

    #[test]
    fn simple_lexing() {
        let terminal_defs = vec![word(), space()];

        let ignore_types = HashSet::from([space()]);

        // Initialize the lexer
        let Ok(lexer) = Lexer::new(terminal_defs, ignore_types) else {
            panic!()
        };

        // Lex a simple text
        let tokens = lexer.lex("hello world".as_bytes()).unwrap();

        // Should have 2 tokens: "hello" and "world"
        // (plus one EOF marker)
        assert_eq!(tokens.0.len(), 2);

        assert_eq!(&*tokens.0[0].value, "hello".as_bytes());
        assert_eq!(tokens.0[0].terminal, Some(word()));

        assert_eq!(&*tokens.0[1].value, "world".as_bytes());
        assert_eq!(tokens.0[1].terminal, Some(word()));

        // The remainder should be the last token in the input.
        assert_eq!(tokens.0[1], tokens.1);
    }

    #[test]
    fn expression() {
        let Ok(lexer) = Lexer::new(
            vec![word(), star(), dec_number(), plus(), space()],
            HashSet::from([space()]),
        ) else {
            panic!()
        };

        let input = "A * 2 + 1".as_bytes();

        let Ok((_tokens, remainder)) = lexer.lex(input) else {
            panic!()
        };
        assert_eq!(
            Token {
                value: "1".as_bytes().into(),
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
    }

    #[test]
    fn complex_string_literals() {
        let terminal_defs = vec![string(), word(), equals(), dot(), space()];

        let ignore_types = HashSet::from([space()]);

        let Ok(lexer) = Lexer::new(terminal_defs, ignore_types) else {
            panic!()
        };

        // Test a simple triple-quoted string
        let text = r#"x = """This is a simple string"""."#.as_bytes();
        let tokens = lexer.lex(text).unwrap();

        // Extract token types
        let token_types: Vec<Terminal> = tokens
            .0
            .iter()
            .map(|token| token.terminal.clone().unwrap())
            .collect();

        // Expected: WORD, EQUALS, STRING, DOT
        assert_eq!(token_types, vec![word(), equals(), string(), dot()]);
    }

    #[test]
    fn numeric_literals() {
        let terminal_defs = vec![
            float_number(),
            hex_number(),
            oct_number(),
            bin_number(),
            dec_number(),
            word(),
            equals(),
            semicolon(),
            space(),
        ];

        let ignore_types = HashSet::from([space()]);

        // Test cases for numeric literals
        let test_cases = vec![
            (
                "x = 42;",
                vec![
                    (word(), "x".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (dec_number(), "42".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "hex = 0xFF;",
                vec![
                    (word(), "hex".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (hex_number(), "0xFF".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "oct = 0o77;",
                vec![
                    (word(), "oct".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (oct_number(), "0o77".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "bin = 0b1010;",
                vec![
                    (word(), "bin".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (bin_number(), "0b1010".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "pi = 3.14159;",
                vec![
                    (word(), "pi".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (float_number(), "3.14159".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "e = 2.71e-3;",
                vec![
                    (word(), "e".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (float_number(), "2.71e-3".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "val = .5;",
                vec![
                    (word(), "val".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (float_number(), ".5".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
            (
                "sci = 6.022e23;",
                vec![
                    (word(), "sci".as_bytes()),
                    (equals(), "=".as_bytes()),
                    (float_number(), "6.022e23".as_bytes()),
                    (semicolon(), ";".as_bytes()),
                ],
            ),
        ];

        let Ok(lexer) = Lexer::new(terminal_defs, ignore_types) else {
            panic!()
        };

        for (text, expected_tokens) in test_cases {
            let tokens = lexer.lex(text.as_bytes()).unwrap();

            // Check token types and values (excluding EOF)
            let token_info: Vec<(Terminal, &[u8])> = tokens
                .0
                .iter()
                .map(|token| (token.terminal.clone().unwrap(), &*token.value))
                .collect();

            assert_eq!(token_info, expected_tokens, "Failed for text: {}", text);
        }
    }

    #[test]
    fn remainder_is_lexical_token() {
        // Example from page 10 of the paper. In the case where the string
        // could be lexed all the way to the end, the remainder is the last
        // lexical terminal (because that could change its type with future
        // additions).
        let terminals = vec![word(), dec_number(), space()];

        let ignore_types = HashSet::from([space()]);

        let Ok(lexer) = Lexer::new(terminals, ignore_types) else {
            panic!()
        };

        let text = "123 ret".as_bytes();
        let (tokens, remainder) = lexer.lex(text).unwrap();

        // We expect:
        // tokens: [123, ret]
        // remainder: ret
        assert_eq!(
            tokens[0],
            Token {
                value: "123".as_bytes().into(),
                terminal: Some(dec_number()),
                start_pos: 0,
                end_pos: 3,
                line: 1,
                column: 1,
                end_line: 1,
                end_column: 4
            }
        );

        assert_eq!(
            tokens[1],
            Token {
                value: "ret".as_bytes().into(),
                terminal: Some(word()),
                start_pos: 4,
                end_pos: 7,
                line: 1,
                column: 5,
                end_line: 1,
                end_column: 8
            }
        );

        assert_eq!(tokens[1], remainder);
    }

    #[test]
    fn remainder_is_not_lexical_token() {
        // In the case where the string could not be lexed all the way to the
        // end, the remainder is unlexed suffix.
        let terminals = vec![word(), hex_number(), space()];

        let ignore_types = HashSet::from([space()]);

        let Ok(lexer) = Lexer::new(terminals, ignore_types) else {
            panic!()
        };

        let text = "return 0x".as_bytes();
        let (tokens, remainder) = lexer.lex(text).unwrap();

        // We expect:
        // tokens: [return]
        // remainder: 0x
        assert_eq!(
            tokens[0],
            Token {
                value: "return".as_bytes().into(),
                terminal: Some(word()),
                start_pos: 0,
                end_pos: 6,
                line: 1,
                column: 1,
                end_line: 1,
                end_column: 7
            }
        );

        assert_eq!(
            remainder,
            Token {
                value: "0x".as_bytes().into(),
                terminal: None,
                start_pos: 7,
                end_pos: 9,
                line: 1,
                column: 8,
                end_line: usize::MAX,
                end_column: usize::MAX
            }
        );
    }

    #[test]
    fn multiline_tracking() {
        let terminal_defs = vec![word(), newline(), space()];

        let ignore_types = HashSet::from([space()]);

        let Ok(lexer) = Lexer::new(terminal_defs, ignore_types) else {
            panic!()
        };

        // Test multiline text
        let text = "first\nsecond\nthird".as_bytes();

        let tokens = lexer.lex(text).unwrap();

        // Check line numbers
        assert_eq!(tokens.0.len(), 5); // 3 words + 2 newlines

        // First word should be on line 1
        assert_eq!(tokens.0[0].line, 1);
        assert_eq!(tokens.0[0].value, "first".as_bytes().into());

        // After first newline, we should be on line 2
        assert_eq!(tokens.0[2].line, 2);
        assert_eq!(tokens.0[2].value, "second".as_bytes().into());

        // After second newline, we should be on line 3
        assert_eq!(tokens.0[4].line, 3);
        assert_eq!(tokens.0[4].value, "third".as_bytes().into());

        // The remainder should be the last token seen.
        assert_eq!(tokens.1, tokens.0[4]);
    }
}
