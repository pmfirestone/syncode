// src/grammar.rs
//! Parse [Lark's EBNF](https://lark-parser.readthedocs.io/en/stable/grammar.html)
//! and turn it into a `[crate::types::Grammar]` object.
//!
//! The comments of this file make liberal use of text from Lark's grammar and
//! documentation, without explicit attribution.
//!
//! FIXMES:
//! 
//! 1. There are `consume_space()`s throughout the module sort of at
//! random. Should this behavior be integrated directly into the consume
//! procedure, so that we never have to explicitly worry about it?
//! 
//! 2. Is it ideal to keep the input string and a current position index
//! seperately in the module's state, rather than just consuming characters
//! from an iterator? The position index allows for backtracking, but it's not
//! clear that we ever need that.
use crate::types::*;
use regex::Regex;
use regex_automata::util::lazy::Lazy;

/// Track the current state of parsing.
///
/// Only one of these structs will exist throughout the life of the program,
/// and it will be mutated a lot.
struct EBNFParser {
    /// The current position in the input.
    cur_pos: usize,
    /// The name of the starting rule in the grammar.
    starting_rule_name: Box<str>,
    /// The ebnf grammar that we are currently parsing.
    input_string: Box<str>,
    /// The grammar that we will eventually return.
    grammar: Grammar,
    /// The current line.
    cur_line: usize,
    /// The current column.
    cur_column: usize,
    /// The name of the rule we are currently parsing (i.e. the nonterminal on
    /// the production's left-hand side).
    cur_rule_name: Box<str>,
    /// The name of the token we are currently parsing. "" if none.
    cur_token_name: Box<str>,
    /// The priority of the rule or token we are currently parsing. Defaults to 0.
    cur_priority: i32,
    /// Whether or not we are currently inside parentheses. "" if none.
    in_parentheses: bool,
}

/// The terminals of the grammar description language.

const STRING: &str = r#"\".*?(?<!\)(\\)*?\"i?"#;
const REGEXP: &str = r"/(?!/)(\/|\\|[^/])*?/[imslux]*";
const NL: &str = r"(\r?\n)+\s*";

// Anchor these to the beginning of the string, because we will be using the regex
// to pop the terminal off the beginning of the input.
const OP: &str = r"^[+*]|[?](?![a-z])";
const OP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(OP).unwrap());
const VBAR: &str = r"^((\r?\n)+\s*)?\|";
const VBAR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(VBAR).unwrap());
const NUMBER: &str = r"^(+|-)?[0-9]+";
const NUMBER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(NUMBER).unwrap());
const RULE: &str = r"^!?[_?]?[a-z][_a-z0-9]*";
const RULE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(RULE).unwrap());
const TOKEN: &str = r"^_?[A-Z][_A-Z0-9]*";
const TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(TOKEN).unwrap());

impl EBNFParser {
    fn new(input_string: &str, starting_rule_name: &str) -> Self {
        EBNFParser {
            cur_pos: 0,
            starting_rule_name: starting_rule_name.into(),
            input_string: input_string.into(),
            grammar: Grammar {
                productions: vec![],
                terminals: vec![],
                symbol_set: vec![],
                start_production: Production {
                    lhs: "".into(),
                    rhs: vec![],
                    priority: 0,
                },
            },
            cur_line: 1,
            cur_column: 1,
            cur_rule_name: "".into(),
            cur_token_name: "".into(),
            cur_priority: 0,
            in_parentheses: false,
        }
    }
    /// Parse inout_string into a Grammar.
    ///
    /// The grammar will be "augmented", which means that the start rule will
    /// always be a production whose right-hand side is a single non-terminal. This
    /// is necessary for the algorithm in `[crate::table]`, which assumes this
    /// characteristic.
    fn parse(mut self) -> Grammar {
        // Preprocessing.
        self.expand_templates();
        self.expand_imports();
        self.expand_overrides();
        self.expand_extends();

        while self.cur_pos <= self.input_string.len() {
            // Advance to next non-whitespace (or comment) in the input.
            self.consume_space();
            // item: rule | token | statement
            if RULE_RE.is_match_at(&self.input_string, self.cur_pos) {
                self.parse_rule();
            } else if TOKEN_RE.is_match_at(&self.input_string, self.cur_pos) {
                self.parse_token();
            } else if self.input_string[self.cur_pos..self.cur_pos + 1] == *"%" {
                // FIXME: terrible kludge to check whether the next character is "%".
                self.parse_statement();
            } else {
                self.report_parse_error("Invalid item.");
            }
        }
        self.grammar
    }

    /// Parse a rule.
    ///
    /// `rule: RULE priority? ":" expansions`
    ///
    /// Note that the params that are present in the syntax have already been
    /// expanded by the time this procedure is called.
    fn parse_rule(&mut self) {
        let rule_match = RULE_RE.find_at(&self.input_string, self.cur_pos).unwrap();

        self.cur_rule_name = rule_match.as_str().into();
        self.consume(rule_match.len());

        self.consume_space();

        if self.peek(0) == '.' {
            self.parse_priority();
        }

        if self.peek(0) == ':' {
            self.consume(1);
        } else {
            self.report_parse_error("Expected ':'.");
        }

        self.parse_expansions();

        // We are no longer parsing a rule.
        self.cur_rule_name = "".into();
        self.cur_priority = 0;
    }

    /// Parse a token.
    ///
    /// `token: TOKEN priority? ":" expansions`
    ///
    /// Note that the params that are present in the syntax have already been
    /// expanded by the time this procedure is called.
    fn parse_token(&mut self) {
        let token_match = TOKEN_RE.find_at(&self.input_string, self.cur_pos).unwrap();

        self.cur_token_name = token_match.as_str().into();
        self.consume(token_match.len());

        if self.peek(0) == '.' {
            self.parse_priority();
        }

        if self.peek(0) == ':' {
            self.consume(1);
        } else {
            self.report_parse_error("Expected ':'.");
        }

        self.parse_expansions();

        // We are no longer parsing a token.
        self.cur_token_name = "".into();
        self.cur_priority = 0;
    }

    /// Parse a statement of the grammar.
    ///
    /// ```
    /// statement: "%ignore" expansions                    -> ignore
    ///          | "%declare" name+                        -> declare
    /// ```
    ///
    /// Note that %import, %override, and %extend directives will have already
    /// been expanded by the preprocessor by the time this procedure is called.
    fn parse_statement(&mut self) {
        match &self.input_string[self.cur_pos..self.cur_pos + 3] {
            // 3 characters are sufficient to disambiguate.
            "%ig" => {
                self.consume("%ignore".len());
                self.parse_ignore()
            }
            "%de" => {
                self.consume("%declare".len());
                self.parse_declare()
            }
            _ => self.report_parse_error("Invalid statement."),
        }
    }

    /// Parse the priority.
    ///
    /// `priority: "." NUMBER`
    ///
    /// A priority can only be followed by a ":".
    fn parse_priority(&mut self) {
        if self.peek(0) == '.' {
            self.consume(1);
        } else {
            self.report_parse_error("Expected '.'.");
        }

        let Some(number_match) = NUMBER_RE.find_at(&self.input_string, self.cur_pos) else {
            // Add a return statement to make the compiler happy, even though
            // this call never returns.
            return self.report_parse_error("Expected a number.");
        };

        // Parsing as an integer *should* never fail, since we've just matched
        // something that must be a valid number.
        self.cur_priority = number_match.as_str().parse::<i32>().unwrap();
        self.consume(number_match.len());
    }

    /// Parse the expansions of the rule.
    ///
    /// `?expansions: alias (_VBAR alias)*`
    ///
    // This begins a vertiginous descent into the bowels of the grammar.
    fn parse_expansions(&mut self) {
        self.consume_space();
        self.parse_alias();
        self.consume_space();
        while VBAR_RE.is_match_at(&self.input_string, self.cur_pos) {
            self.consume_space();
            self.parse_alias();
        }
    }

    /// Parse an alias.
    ///
    /// `?alias: expansion ["->" RULE]`
    fn parse_alias(&mut self) {
        self.consume_space();
        self.parse_expansion();
        self.consume_space();
        if self.input_string[self.cur_pos..self.cur_pos + 2] == *"->" {
            // We don't actually vare about aliases: they only matter to the
            // parse tree that Lark would build if it was using this
            // grammar. Just skip to the end of the line.
            self.consume_to_end_of_line();
        }
    }

    /// Parse an expansion.
    ///
    /// `?expansion: expr*`
    ///
    /// A Lark expansion must be on a single line.
    fn parse_expansion(&mut self) {
        // Naughty way to check whether we're still on the same line.
        let start_line = self.cur_line;
        while self.cur_line == start_line {
            self.consume_space();
            self.parse_expression();
        }
    }

    /// Parse an expression.
    ///
    /// ?expr: atom (OP | "~" NUMBER (".." NUMBER))?
    ///
    /// This level of the grammar separates the quantifiers from the thing they
    /// quantify.
    fn parse_expression(&mut self) {
        self.parse_atom();
	self.consume_space();
	if OP_RE.is_match_at(&self.input_string, self.cur_pos) {
	    self.handle_op();
	} else if self.peek(0) == '~' {
	    self.consume(1);
	    self.consume_space();
	    self.handle_range();
	}
    }

    /// Expand the templates in the grammar.
    ///
    /// Definition syntax:
    /// `my_template{param1, param2, ...}: <EBNF EXPRESSION>`
    /// Use syntax:
    /// `some_rule: my_template{arg1, arg2, ...}`
    /// Example:
    /// ```
    /// _separated{x, sep}: x (sep x)*  // Define a sequence of 'x sep x sep x ...'
    /// num_list: "[" _separated{NUMBER, ","} "]"   // Will match "[1, 2, 3]" etc.
    ///  ```
    /// This is a macro expander that modifies the input string in place.
    fn expand_templates(&mut self) {}

    /// Expand the imports in the grammar.
    ///
    /// When importing rules, all their dependencies will be imported into a
    /// namespace, to avoid collisions. It’s not possible to override their
    /// dependencies (e.g. like you would when inheriting a class).
    ///
    /// Syntax:
    ///    ```
    /// %import <module>.<TERMINAL>
    /// %import <module>.<rule>
    /// %import <module>.<TERMINAL> -> <NEWTERMINAL>
    /// %import <module>.<rule> -> <newrule>
    /// %import <module> (<TERM1>, <TERM2>, <rule1>, <rule2>)
    /// ```
    ///
    /// If the module path is absolute, [we] will attempt to load it from the
    /// built-in directory (which currently contains common.lark, python.lark,
    /// and unicode.lark).
    ///
    /// If the module path is relative, such as .path.to.file, [we] will
    /// attempt to load it from the current working directory. Grammars must
    /// have the .lark extension.
    ///
    /// The rule or terminal can be imported under another name with the -> syntax.
    fn expand_imports(&mut self) {}

    /// Expand override directives in the grammar.
    ///
    /// Override a rule or terminals, affecting all references to it, even in imported grammars.
    ///
    /// Useful for implementing an inheritance pattern when importing grammars.
    ///
    /// Example:
    /// ```
    /// %import my_grammar (start, number, NUMBER)
    /// // Add hex support to my_grammar
    /// %override number: NUMBER | /0x\w+/
    /// ````
    ///
    /// There is no requirement for a rule/terminal to come from another file,
    /// but that is probably the most common use case.
    fn expand_overrides(&mut self) {}

    /// Expand extend directives in the grammar.
    ///
    /// Extend the definition of a rule or terminal, e.g. add a new option on
    /// what it can match, like when separated with |.
    ///
    /// Useful for splitting up a definition of a complex rule with many
    /// different options over multiple files.
    ///
    /// Can also be used to implement a plugin system where a core grammar is extended by others.
    ///
    /// Example:
    /// ```
    /// %import my_grammar (start, NUMBER)
    ///
    /// // Add hex support to my_grammar
    /// %extend NUMBER: /0x\w+/
    /// ```
    ///
    /// There is no requirement for a rule/terminal to come from another file,
    /// but that is probably the most common use case.
    fn expand_extends(&mut self) {}

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

    /// Consume the next whitespace and comment in the input.
    fn consume_space(&mut self) {
        while self.cur_pos < self.input_string.len()
            && (self.peek(0) == ' '
                || self.peek(0) == '\t'
                || self.peek(0) == '#'
                || self.peek(0) == '/' && self.peek(1) == '/')
        {
            if self.peek(0) == '#' || self.peek(0) == '/' && self.peek(1) == '/' {
                self.consume_to_end_of_line();
            }
        }
    }

    /// Consume to the end of the line.
    ///
    /// Useful for skipping aliases and comments.
    fn consume_to_end_of_line(&mut self) {
        while self.cur_pos < self.input_string.len() {
            if !(self.peek(0) == '\n' || (self.peek(0) == '\r' && self.peek(0) == '\n')) {
                self.consume(1);
            } else {
                break;
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
