// src/lexer.rs
//! The lexer for SynCode. The primary procedure for this module is `lex`
//! (q.v.), which takes a text and returns a sequence of lexical tokens along
//! with a "remainder". See the paper for more detail.
use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::{Anchored, util::start};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A lexical token, what the lexer breaks the input into.
#[derive(Clone, Debug, PartialEq)]
pub struct Token<'a> {
    /// The content of the token.
    pub value: &'a str,
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

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({:?}, {})", self.terminal, self.value)
    }
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

impl fmt::Display for Terminal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Terminal({}, {}, {})",
            self.name, self.pattern, self.priority
        )
    }
}

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
struct Scanner<'a> {
    /// The DFA for matching patterns.
    dfa: dense::DFA<Vec<u32>>,
    /// Maps DFA match pattern to the TerminalDef it represents.
    index_to_type: HashMap<usize, Terminal<'a>>,
    /// Maps token type name to whether it can contain newlines.
    newline_types: HashSet<&'a str>,
    /// Terminal definitions for reference.
    terminals: Vec<Terminal<'a>>,
    /// All allowed types.
    pub allowed_types: HashSet<&'a str>,
}

impl<'a> Scanner<'a> {
    pub fn new(terminals: Vec<Terminal<'a>>) -> Result<Self, LexError> {
        let mut newline_types = HashSet::new();
        let mut allowed_types = HashSet::with_capacity(terminals.len());
        let mut index_to_type = HashMap::with_capacity(terminals.len());

        // Determine which patterns might contain newlines
        for terminal in &terminals {
            let pattern_str = terminal.pattern.clone();
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
            patterns.push(terminal.pattern);
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
            newline_types,
            terminals: sorted_terminals,
            allowed_types,
        })
    }

    /// Match the next token in the input, beginning at position pos, and
    /// return it along with the type of terminal that it is.
    ///
    /// Look for the longest possible match.
    pub fn match_token(&self, text: &'a str, pos: usize) -> Option<(&'a str, &Terminal)> {
        if pos >= text.len() {
            return None;
        }

        let rest = &text[pos..];
        let bytes = rest.as_bytes();

        let config = start::Config::new().anchored(Anchored::Yes);
        let mut state = self.dfa.start_state(&config).expect("no look-around");

        if self.dfa.is_dead_state(state) {
            return None;
        }

        // Keep track of the best match so far
        let mut best_match: Option<(usize, usize)> = None; // (pattern_idx, length)
        let mut current_len = 0;

        // Walk through the DFA state by state
        for &byte in bytes {
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
                return Some((&rest[..match_len], terminal));
            }
        }

        None
    }
}

/// A lexer.
#[derive(Clone)]
pub struct Lexer<'a> {
    /// The machinery for the DFAs.
    scanner: Option<Scanner<'a>>,
    /// The terminals this lexer recognizes.
    pub terminals: Vec<Terminal<'a>>,
    /// The terminals that this lexer ignores.
    pub ignore_types: HashSet<Terminal<'a>>,
    /// The terminals that contain newlines.
    pub newline_types: HashSet<Terminal<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new() -> Self {
        Lexer {
            scanner: None,
            terminals: Vec::new(),
            ignore_types: HashSet::new(),
            newline_types: HashSet::new(),
        }
    }

    pub fn initialize(
        &mut self,
        terminals: Vec<Terminal<'a>>,
        ignore_types: HashSet<Terminal<'a>>,
    ) -> Result<(), LexError> {
        self.ignore_types = ignore_types;

        // Determine which patterns might contain newlines
        for terminal in &terminals {
            if terminal.pattern.contains("\\n")
                || terminal.pattern.contains("\n")
                || terminal.pattern.contains("\\s")
                || terminal.pattern.contains("[^")
                || (terminal.pattern.contains(".") && terminal.pattern.contains("(?s"))
            {
                self.newline_types.insert(terminal.clone());
            }
        }

        // Create scanner
        match Scanner::new(terminals.clone()) {
            Ok(scanner) => {
                self.scanner = Some(scanner);
                self.terminals = terminals;
                Ok(())
            }
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
        &'a self,
        text: &'a str,
        mut pos: usize,
        mut line: usize,
        mut column: usize,
        //        last_token: Option<&Token>,
    ) -> Result<(Token<'a>, bool), LexError> {
        // Ensure scanner is initialized
        let scanner = match &self.scanner {
            Some(s) => s,
            None => {
                return Err(LexError::InitError(
                    "Scanner not initialized. Call initialize() first.".to_string(),
                ));
            }
        };

        loop {
            // Try to match next token
            if let Some((value, terminal)) = scanner.match_token(text, pos) {
                let ignored = self.ignore_types.contains(terminal);

                // If this token is ignored, update position and continue the loop
                if ignored {
                    let contains_newline = self.newline_types.contains(terminal);

                    // Update line and column information
                    if contains_newline {
                        // Calculate new line and column for tokens with newlines
                        for c in value.chars() {
                            if c == '\n' {
                                line += 1;
                                column = 1;
                            } else {
                                column += 1;
                            }
                        }
                    } else {
                        column += value.chars().count();
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

                    for c in value.chars() {
                        if c == '\n' {
                            current_line += 1;
                            current_column = 1;
                        } else {
                            current_column += 1;
                        }
                    }

                    (current_line, current_column)
                } else {
                    // Simple calculation for tokens without newlines
                    (line, column + value.chars().count())
                };

                return Ok((
                    Token {
                        value,
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
                        value,
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
    pub fn lex(&'a self, text: &'a str) -> Result<(Vec<Token<'a>>, Token<'a>), LexError> {
        if self.scanner.is_none() {
            return Err(LexError::InitError(
                "Scanner not initialized. Call initialize() first.".to_string(),
            ));
        }

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
    const WORD: Terminal = Terminal {
        name: "WORD",
        pattern: r"[a-zA-Z_]\w*",
        priority: 2,
    };

    const STRING: Terminal = Terminal {
        name: "STRING",
        pattern: r#"("""[^"]*"""|'''[^']*''')"#,
        priority: 2,
    };

    const SPACE: Terminal = Terminal {
        name: "SPACE",
        pattern: "\\s+",
        priority: 0,
    };

    const EQUALS: Terminal = Terminal {
        name: "EQUALS",
        pattern: "=",
        priority: 1,
    };

    const DOT: Terminal = Terminal {
        name: "DOT",
        pattern: r"\.",
        priority: 1,
    };

    const DEC_NUMBER: Terminal = Terminal {
        name: "DEC_NUMBER",
        pattern: r"0|[1-9]\d*",
        priority: 1,
    };

    const OCT_NUMBER: Terminal = Terminal {
        name: "OCT_NUMBER",
        pattern: r"(?i)0o[0-7]+",
        priority: 1,
    };

    const BIN_NUMBER: Terminal = Terminal {
        name: "BIN_NUMBER",
        pattern: r"(?i)0b[0-1]+",
        priority: 1,
    };

    const HEX_NUMBER: Terminal = Terminal {
        name: "HEX_NUMBER",
        pattern: r"(?i)0x[\da-f]+",
        priority: 1,
    };

    const FLOAT_NUMBER: Terminal = Terminal {
        name: "FLOAT_NUMBER",
        pattern: r"((\d+\.\d*|\.\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))",
        priority: 1,
    };

    const SEMICOLON: Terminal = Terminal {
        name: "SEMICOLON",
        pattern: ";",
        priority: 0,
    };

    const NEWLINE: Terminal = Terminal {
        name: "NEWLINE",
        pattern: r"\n",
        priority: 1,
    };

    #[test]
    fn lexer_initialization() {
        let mut lexer = Lexer::new();

        let terminal_defs = vec![WORD, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Check if it was initialized correctly
        assert!(lexer.scanner.is_some());
        assert_eq!(lexer.terminals.len(), 2);
        assert_eq!(lexer.ignore_types.len(), 1);
    }

    #[test]
    fn simple_lexing() {
        let mut lexer = Lexer::new();

        let terminal_defs = vec![WORD, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Lex a simple text
        let tokens = lexer.lex("hello world").unwrap();

        // Should have 2 tokens: "hello" and "world"
        // (plus one EOF marker)
        assert_eq!(tokens.0.len(), 2);

        assert_eq!(tokens.0[0].value, "hello");
        assert_eq!(tokens.0[0].terminal, Some(WORD));

        assert_eq!(tokens.0[1].value, "world");
        assert_eq!(tokens.0[1].terminal, Some(WORD));

        // The remainder should be the last token in the input.
        assert_eq!(tokens.0[1], tokens.1);
    }

    #[test]
    fn complex_string_literals() {
        let mut lexer = Lexer::new();

        let terminal_defs = vec![STRING, WORD, EQUALS, DOT, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Test a simple triple-quoted string
        let text = r#"x = """This is a simple string"""."#;
        let tokens = lexer.lex(text).unwrap();

        // Extract token types
        let token_types: Vec<Terminal> = tokens
            .0
            .iter()
            .map(|token| token.terminal.clone().unwrap())
            .collect();

        // Expected: WORD, EQUALS, STRING, DOT
        assert_eq!(token_types, vec![WORD, EQUALS, STRING, DOT]);
    }

    #[test]
    fn numeric_literals() {
        let terminal_defs = vec![
            FLOAT_NUMBER,
            HEX_NUMBER,
            OCT_NUMBER,
            BIN_NUMBER,
            DEC_NUMBER,
            WORD,
            EQUALS,
            SEMICOLON,
            SPACE,
        ];

        let ignore_types = HashSet::from([SPACE]);

        // Test cases for numeric literals
        let test_cases = vec![
            (
                "x = 42;",
                vec![
                    (WORD, "x"),
                    (EQUALS, "="),
                    (DEC_NUMBER, "42"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "hex = 0xFF;",
                vec![
                    (WORD, "hex"),
                    (EQUALS, "="),
                    (HEX_NUMBER, "0xFF"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "oct = 0o77;",
                vec![
                    (WORD, "oct"),
                    (EQUALS, "="),
                    (OCT_NUMBER, "0o77"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "bin = 0b1010;",
                vec![
                    (WORD, "bin"),
                    (EQUALS, "="),
                    (BIN_NUMBER, "0b1010"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "pi = 3.14159;",
                vec![
                    (WORD, "pi"),
                    (EQUALS, "="),
                    (FLOAT_NUMBER, "3.14159"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "e = 2.71e-3;",
                vec![
                    (WORD, "e"),
                    (EQUALS, "="),
                    (FLOAT_NUMBER, "2.71e-3"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "val = .5;",
                vec![
                    (WORD, "val"),
                    (EQUALS, "="),
                    (FLOAT_NUMBER, ".5"),
                    (SEMICOLON, ";"),
                ],
            ),
            (
                "sci = 6.022e23;",
                vec![
                    (WORD, "sci"),
                    (EQUALS, "="),
                    (FLOAT_NUMBER, "6.022e23"),
                    (SEMICOLON, ";"),
                ],
            ),
        ];

        for (text, expected_tokens) in test_cases {
            // Make a new lexer every time through this loop to make the compiler happy.
            let mut lexer = Lexer::new();
            lexer
                .initialize(terminal_defs.clone(), ignore_types.clone())
                .unwrap();
            let tokens = lexer.lex(text).unwrap();

            // Check token types and values (excluding EOF)
            let token_info: Vec<(Terminal, &str)> = tokens
                .0
                .iter()
                .map(|token| (token.terminal.clone().unwrap(), token.value))
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
        let terminals = vec![WORD, DEC_NUMBER, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        let mut lexer = Lexer::new();
        lexer.initialize(terminals, ignore_types).unwrap();

        let text = "123 ret";
        let (tokens, remainder) = lexer.lex(text).unwrap();

        // We expect:
        // tokens: [123, ret]
        // remainder: ret
        assert_eq!(
            tokens[0],
            Token {
                value: "123",
                terminal: Some(DEC_NUMBER),
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
                value: "ret",
                terminal: Some(WORD),
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
        let terminals = vec![WORD, HEX_NUMBER, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        let mut lexer = Lexer::new();
        lexer.initialize(terminals, ignore_types).unwrap();

        let text = "return 0x";
        let (tokens, remainder) = lexer.lex(text).unwrap();

        // We expect:
        // tokens: [return]
        // remainder: 0x
        assert_eq!(
            tokens[0],
            Token {
                value: "return",
                terminal: Some(WORD),
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
                value: "0x",
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
        let mut lexer = Lexer::new();

        let terminal_defs = vec![WORD, NEWLINE, SPACE];

        let ignore_types = HashSet::from([SPACE]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Test multiline text
        let text = "first\nsecond\nthird";

        let tokens = lexer.lex(text).unwrap();

        // Check line numbers
        assert_eq!(tokens.0.len(), 5); // 3 words + 2 newlines

        // First word should be on line 1
        assert_eq!(tokens.0[0].line, 1);
        assert_eq!(tokens.0[0].value, "first");

        // After first newline, we should be on line 2
        assert_eq!(tokens.0[2].line, 2);
        assert_eq!(tokens.0[2].value, "second");

        // After second newline, we should be on line 3
        assert_eq!(tokens.0[4].line, 3);
        assert_eq!(tokens.0[4].value, "third");

        // The remainder should be the last token seen.
        assert_eq!(tokens.1, tokens.0[4]);
    }
}
