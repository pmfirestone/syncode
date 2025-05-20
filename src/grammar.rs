// src/grammar.rs
//! Parse [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
//! files and turn them into `[crate::types::Grammar]` objects for the
//! `[crate::table]` module to build an LR parser out of.
//!
//! This module implements a recursive-descent parser for GBNF and is
//! essentially a translation into rust of the equivalent component of XGrammar
//! (grammar_parser.cc). GBNF, unfortunately, lacks a formal specification, so
//! this parser is something of a folk theorem whose correctness remains to be shown.
//!
//! GBNF is also strange in that it doesn't actually have regex-based lexical
//! terminals. This makes it somewhat different from other grammar description
//! langauges, where the lexer is separately defined with its own regexes. It
//! may be that this breaks the assumptions of syncode, or not. In principle,
//! one doesn't need the regexes apart from the grammar, since the grammar
//! itself suffices to define the terminals in question. The primary change
//! would be that the lexical tokens will be smaller and the parse tree
//! larger. For example, if an identifier is defined as
//! /[a-zA-Z_][a-zA-Z0-9_]*/, the first character and the possible continuation
//! will be clustered into a single terminal by the lexer. If, instead, an
//! identifier is a production (identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*, the
//! lexer will return the first character then the remaining characters as two
//! separate terminals, and the parser will be responsible for reducing them to
//! the identifier nonterminal. At a first approximation, this should not
//! affect syncode's correctness, since it does not change the inputs that will
//! or won't be recognized by the grammar; in practice, though, the algorithm
//! will examine less of each token, potentially reducing correctness.
use crate::types::*;

/// Track the current state of parsing.
///
/// Only one of these structs will exist throughout the life of the program,
/// and it will be mutated a lot.
struct EBNFParser {
    /// The current position in the input.
    cur_pos: usize,
    /// The name of the starting rule in the grammar.
    starting_rule_name: &'static str,
    /// The gbnf grammar that we are currently parsing. This is a String to
    /// support the presence of non-ascii characters (i.e. multi-byte utf-8
    /// characters) in the grammar specification.
    input_string: String,
    /// The grammar that we will eventually return.
    grammar: Grammar,
    /// The current line.
    cur_line: usize,
    /// The current column.
    cur_column: usize,
    /// The name of the rule we are currently parsing (i.e. the nonterminal on
    /// the production's left-hand side).
    cur_rule_name: &'static str,
    /// Whether or not we are currently inside parentheses.
    in_parentheses: bool,
}

impl EBNFParser {
    fn new(input_string: String, starting_rule_name: &'static str) -> Self {
        EBNFParser {
            cur_pos: 0,
            starting_rule_name,
            input_string,
            grammar: Grammar {
                productions: vec![],
                symbol_set: vec![],
                start_production: Production {
                    lhs: "".into(),
                    rhs: vec![],
                },
            },
            cur_line: 1,
            cur_column: 1,
            cur_rule_name: "",
            in_parentheses: false,
        }
    }
    /// Parse ebnf_string into a Grammar.
    ///
    /// The grammar will be "augmented", which means that the start rule will
    /// always be a production whose right-hand side is a single non-terminal. This
    /// is necessary for the algorithm in `[crate::table]`, which assumes this
    /// characteristic.
    fn parse(mut self) -> Grammar {
        // Just to be sure that nothing silly's happened.
        self.cur_pos = 0;

        self.consume_space(true);
        while self.cur_pos < self.input_string.len() {
            // Throw an error when there are multiple lookahead assertions.
            if self.peek(0) == '(' && self.peek(1) == '=' {
                self.report_parse_error("Unexpected lookahead assertion");
            }
            let new_rules = self.parse_rule();
        }

        self.grammar
    }

    fn parse_identifier(&mut self, accept_empty: bool) -> &str {
        let start = self.cur_pos;
        let mut first_char = true;
        while self.cur_pos <= self.input_string.len() && self.is_name_char(self.peek(0), first_char)
        {
            self.consume(1);
            first_char = false;
        }
        if start == self.cur_pos && !accept_empty {
            self.report_parse_error("Expect rule name");
        }
        return &self.input_string[start..self.cur_pos];
    }

    /// Parse a character class.
    ///
    /// This procedure will in fact always return a Terminal. We get this for
    /// cheap by just passing the character class directly to the regex
    /// crate. Lacking a better option in GBNF, where terminals are defined
    /// indirectly within the right-hand sides of productions, rather than
    /// explicitly as part of a lexer specification, just make the name of the
    /// character class terminal the regex itself.
    fn parse_character_class(&mut self) -> Symbol {
        // We have to scan forward until we get to the end of the part that's
        // the character class, accounting for the optional repeat operators we
        // might find. We want to forward all parsing of these classes to the
        // regex library; all we do here is extract the part of the string we
        // want.
        let mut chars: String = "".to_string();
        while self.cur_pos <= self.input_string.len()
	    // We don't want to terminate for escaped ']'.
            && !(self.peek(0) == '\\' && self.peek(1) == ']')
	    && self.peek(1) != ']'
        {
            chars.push(self.peek(0));
            self.consume(1);
        }
        chars.push(']');
        // A sanity check to make me feel better.
        assert!(
            chars.chars().nth(0).unwrap() == '[',
            "Wrong opening to character class string: {}",
            chars.chars().nth(0).unwrap()
        );
        assert!(
            chars.chars().last().unwrap() == ']',
            "Wrong ending to character class string: {}",
            chars.chars().last().unwrap()
        );
        // We've gotten to the end of the character class proper, but there could be repeat markers.
        match self.peek(0) {
            '*' | '+' | '?' => {
                chars.push(self.peek(0));
                self.consume(1)
            }
            '{' => {
		// Consume to the closing brace.
		while self.cur_pos <= self.input_string.len() && self.peek(0) != '}' {
		    chars.push(self.peek(0));
		    self.consume(1);
		}
		chars.push('}');
		self.consume(1);
	    }
            _ => { /* Do nothing.*/ }
        }

        Symbol::Terminal(Terminal::new(&chars, &chars, 0))
    }

    /// Parse a string in the input.
    ///
    /// In GBNF, a string is always a terminal of the grammar, so this
    /// procedure will always return a terminal. In the absence of another way
    /// to determine the name of the terminal, we will simply use the string
    /// itself as its own name.
    ///
    /// We can cheese this by relying on Rust's string abstractions; we don't
    /// have to juggle the utf-8 ourselves, unlike XGrammar.
    fn parse_string(&mut self) -> Symbol {
        let mut chars: String = "".into();
        while self.cur_pos <= self.input_string.len() && self.peek(1) != '"' {
            chars.push(self.peek(0));
            self.consume(1);
        }
        Symbol::Terminal(Terminal::new(&chars, &chars, 0))
    }

    /// Determine whether this character could be part of an identifier.
    fn is_name_char(&mut self, c: char, first_char: bool) -> bool {
        return c == '_'
            || c == '-'
            || c == '.'
            || c.is_ascii_alphabetic()
            || (!first_char && c.is_ascii_digit());
    }

    /// Parse a reference to another rule on the right-hand side of a rule.
    ///
    /// Since a nonterminal is just its own name, this procedure always returns
    /// a nonterminal.
    fn parse_rule_ref(&mut self) -> Symbol {
        let rule_name = self.parse_identifier(false);
        Symbol::NonTerminal(rule_name.into())
    }

    /// Parse a single element of the right-hand side of a production.
    fn parse_element(&mut self) -> Symbol {
        match self.peek(0) {
            '(' => {
                self.consume(1);
                self.consume_space(true);
                if self.peek(0) == ')' {
                    // Special case: ( ).
                    self.consume(1);
                    return Symbol::Terminal(*EPSILON);
                }
                let prev_in_parentheses = self.in_parentheses;
                self.in_parentheses = true;
                let choices = self.parse_choices();
                self.consume_space(true);
                if self.peek(0) != ')' {
                    self.report_parse_error("Expect )");
                }
                self.consume(1);
                self.in_parentheses = prev_in_parentheses;
                return choices;
            }
            '[' => {
                // Let the parse_character_class procedure handle the opening
                // and closing square braces.
                let element = self.parse_character_class();
                // Note that parsing the character class also consumes whatever
                // quantifier follows the character class, if there is any. We
                // do not consume any farther in the input at this point.
                return element;
            }
            '\"' => {
                self.consume(1);
                return self.parse_string();
            }
            _ => {
                if self.is_name_char(self.peek(0), true) {
                    return self.parse_rule_ref();
                }
                self.report_parse_error("Expect element, but got character: {self.peek(0)}");
                // Make the compiler happy, even though the previous call never returns.
                return Symbol::NonTerminal("".into());
            }
        }
    }

    /// Unpack repeating rules into a bunch of individual rules. This is distinct from the ranges that follow character classes 
    fn handle_repetition_range(&mut self, element: Symbol, lower: usize, upper: usize) {
	
    }

    fn handle_star_quantifier(&mut self, element: Symbol) -> Symbol {}

    fn handle_plus_quantifier(&mut self, element: Symbol) -> Symbol {}

    fn handle_question_quantifier(&mut self, element: Symbol) -> Symbol {}

    fn parse_element_with_quantifier(&mut self) -> Symbol {
        let element: Symbol = self.parse_element();
        self.consume_space(self.in_parentheses);
        if self.peek(0) != '*' && self.peek(0) != '+' && self.peek(0) != '?' && self.peek(0) != '{'
        {
            // Not a quantified element.
            return element;
        }

        // Handle repetition range.
        if self.peek(0) == '{' {
            let (lower, upper) = self.parse_repetition_range();
            return self.handle_repetition_range(element, lower, upper);
        }

        // Get the quantifier to parse and advance.
        let quantifier = self.peek(0);
        self.consume(1);

        match quantifier {
            '*' => return self.handle_star_quantifier(element),
            '+' => return self.handle_plus_quantifier(element),
            '?' => return self.handle_question_quantifier(element),
            _ => {
                self.report_parse_error("Unreachable. Failed to match any quantifier.");
                // Make the compiler happy, even though the the previous line
                // will panic, and anyway this branch can't be reached.
                return element;
            }
        }
    }

    /// Parse a sequence of elements in a single choice on the right-hand side of a production.
    fn parse_sequence(&mut self) -> Vec<Symbol> {
        let mut elements: Vec<Symbol> = Vec::new();
        loop {
            elements.push(self.parse_element_with_quantifier());
            if !(self.cur_pos < self.input_string.len()
                && self.peek(0) != '|'
                && self.peek(0) != ')'
                && self.peek(0) != '\n'
                && self.peek(0) != '\r')
            {
                break;
            }
        }
        elements
    }

    /// Parse the choices on the right hand side of a rule, returning a vector
    /// of each possible result.
    fn parse_choices(&mut self) -> Vec<Vec<Symbol>> {
        let mut choices: Vec<Vec<Symbol>> = Vec::new();
        choices.push(self.parse_sequence());
        self.consume_space(true);
        while self.peek(0) == '|' {
            self.consume(1);
            self.consume_space(true);
            choices.push(self.parse_sequence());
            self.consume_space(true);
        }
        choices
    }

    /// Parse a rule.
    ///
    /// The basic format of a production rule in GBNF is `nonterminal ::=
    /// sequence...`, where sequence is some terminals and nonterminals.
    fn parse_rule(&mut self) -> Vec<Production> {
        let rule_name = self.parse_identifier(false);
        self.cur_rule_name = rule_name;
        self.consume_space(true);
        if self.peek(0) != ':' || self.peek(1) != ':' || self.peek(2) != '=' {
            self.report_parse_error("Expect ::=");
        }
        self.consume(3);
        self.consume_space(true);
        // Here we diverge somewhat from XGrammar, since they do some business
        // with tagged productions at the root rule that we don't. (This is, in
        // fact, not documented in their documentation as far as I can tell,
        // but the relevant term to search in the source code is TagDispatch).

        // We also diverge in that they insert the rules directly into their
        // grammar builder, whereas we put them into the Grammar struct, which
        // will in turn be processed by the table module to make the tables the
        // parser module uses.
        let right_hand_sides: Vec<Vec<Symbol>> = self.parse_choices();
        self.consume_space(true);
        // We also don't (yet) support lookaheads in our grammars, but this is
        // the point in XGrammar where they figure out the lookaheads in the
        // rule, if there are any.

        return right_hand_sides
            .into_iter()
            .map(|rhs| Production {
                lhs: rule_name.into(),
                rhs,
            })
            .collect();
    }

    /// Consume the specified number of characters, maintaining line and column number.
    fn consume(&mut self, count: usize) {
        for _ in 0..count {
            // Newline advances line count, except when "\n\r"? This logic is
            // verbatim from xgrammar.
            if self.peek(0) == '\n' || (self.peek(0) == '\r' && self.peek(1) != '\n') {
                self.cur_line += 1;
                self.cur_column = 1;
            } else {
                self.cur_column += 1;
            }
            self.cur_pos += 1;
        }
    }

    /// Peek the character delta ahead of the current one.
    fn peek(&self, delta: usize) -> char {
        // Evil because the Rust str and String abstractions are surprisingly
        // leaky, which makes them substantially harder to work with than in
        // other languages. This gets, strictly speaking, the (input_pos +
        // delta)th character in the input.
        self.input_string.chars().nth(self.cur_pos + delta).unwrap()
    }

    /// Consume the next whitespace in the input.
    fn consume_space(&mut self, allow_newline: bool) {
        while self.cur_pos < self.input_string.len()
            && (self.peek(0) == ' '
                || self.peek(0) == '\t'
                || self.peek(0) == '#'
                || (allow_newline && (self.peek(0) == '\n' || self.peek(0) == '\r')))
        {
            if self.peek(0) == '#' {
                // Skip over comments, which extend to the end of the line.
                while self.cur_pos < self.input_string.len()
                    && self.peek(0) != '\n'
                    && self.peek(0) != '\r'
                {
                    self.consume(1);
                }
                // XGrammar has a check here with the comment "Reserve
                // \n for inline comment". My C++ isn't good enough to
                // understand the conditions under which their Peek()
                // operation, which dereferences a pointer into the string
                // representing the grammar, can return a value that will be
                // coerced to the bool false. Perhaps they're checking whether
                // the end of the input was reached? Any clarification would be
                // greatly appreciated.
                if self.peek(0) == '\r' && self.peek(1) == 'n' {
                    // Handle CRLF newliens.
                    self.consume(2);
                } else {
                    self.consume(1)
                }
            } else {
                self.consume(1);
            }
        }
    }

    /// Report a parse error with the line and column number. This procedure
    /// will panic!() when called.
    fn report_parse_error(&self, message: &str) {
        panic!(
            "GBNF parse error at line {}, column {}: {message}",
            self.cur_line, self.cur_column
        );
    }
}
