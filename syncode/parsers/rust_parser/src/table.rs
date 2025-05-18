// src/table.rs
//! Produce action and goto tables for the use of Syncode in the [`crate::parse`] module.
//!
//! Based on the relevant sections of the Dragon Book, 2e.
//!
//! This whole module is a spaghetti mess for which I'm going to go to
//! hell. I'm sure that someone who understood the relevant algorithms better
//! than I do could restate them in a way that produced better code, but I'm
//! translating them to Rust directly from the book as I understand it. There
//! is work to be done here in cleaning the code and improving its efficiency;
//! right now the goal is functioning code, nothing more, nothing less.

use std::collections::{HashMap, HashSet};

/// A terminal of the grammar.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Terminal<'a> {
    /// The name of this terminal in the grammar.
    name: &'a str,
    /// The regex describing this terminal.
    pattern: &'a str,
    /// This terminal's priority in lexing.
    priority: i32,
}

/// A type alias for nonterminals of the grammar, purely for readability.
pub type NonTerminal<'a> = &'a str;

/// An enumeration for symbols of the grammar, to act as a union type of terminals and nonterminals.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
enum Symbol {
    Terminal(Terminal<'static>),
    NonTerminal(NonTerminal<'static>),
}

/// A convenience terminal representing the empty string.
static EPSILON: Terminal<'static> = Terminal {
    name: "epsilon",
    pattern: "",
    priority: 0,
};

/// A convenience terminal representing the end of the input.
static EOF: Terminal<'static> = Terminal {
    name: "$",
    pattern: "",
    priority: 0,
};

/// A single production of the grammar.
///
/// That this is exactly a single production, so productions that can go to
/// more than one outcome have to be represented by more than one production in
/// the grammar.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Production {
    /// The left hand side of the production.
    source: NonTerminal<'static>,
    /// The right hand side of the production.
    result: Vec<Symbol>,
}

/// A context-free grammar.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Grammar {
    /// The set of symbols that are active in this grammar.
    symbol_set: Vec<Symbol>,
    /// The first production; this one is the augmented one added to the grammar.
    start_production: Production,
    /// The productions that make up this grammar, including the start_production.
    productions: Vec<Production>,
}

/// An item of the item set for LR parsing.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct Item {
    /// The production that this item contains.
    production: Production,
    /// The position of the dot in the result. Invariant: must be in [0, result.len()].
    dot: usize,
    /// The look ahead terminal.
    lookahead: Terminal<'static>,
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

type ActionTable = HashMap<(usize, Terminal<'static>), Action>;
type GotoTable = HashMap<(usize, NonTerminal<'static>), usize>;


fn symbol_first(symbol: &Symbol, grammar: &Grammar) -> HashSet<Terminal<'static>> {
    match symbol {
	// If symbol is a terminal, then first(symbol) = {symbol}.
	Symbol::Terminal(terminal) => return HashSet::from([terminal.clone()]),
	// If symbol is a nonterminal...
	Symbol::NonTerminal(nonterminal) => {
	    let mut first_set = HashSet::new();
	    for production in grammar.productions.clone() {
		// And symbol -> y1y2...yk for some k >= 1,
		if production.source == *nonterminal {
		    for symbol in &production.result {
			// Place the contents of the first set of the resulting
			// symbol into this symbol's first set...
			let first = symbol_first(&symbol, grammar);
			first_set.extend(first.clone().into_iter());
			if production.result == vec![Symbol::Terminal(EPSILON.clone())] {
			    // If symbol -> ϵ is a production, add ϵ to first(symbol).
			    first_set.insert(EPSILON.clone());
			}
			if !first.contains(&EPSILON) {
			    // Keep adding as long as the first sets contain ϵ.
			    return first_set;
			}
		    }
		}
	    }
	    first_set
	},
    }
}

/// Compute the first set of a string, gotten as the right-hand side of some production.
///
/// The algorithm comes from sec. 4.4.2 of the Dragon Book 2e, p. 221:
///
/// We can compute FIRST for any string X_1X_2...Xn as follows. Add to
/// FIRST(X1X2...Xn) all non-𝜖 symbols of FIRST(X1). Also add the non-𝜖 symbols
/// of FIRST(X2) if 𝜖 is in FIRST(X1); the non-𝜖 symbols of FIRST(X2) if 𝜖 is
/// in FIRST(X1) and FIRST(X2) and so on. Finally add to FIRST(X1X2...Xn) if
/// for all i 𝜖 is in FIRST(Xi).
fn string_first(string: Vec<Symbol>, grammar: &Grammar) -> HashSet<Terminal<'static>> {
    let mut first_set: HashSet<Terminal<'static>> = HashSet::new();
    let string_length = string.len();
    for (idx, outer_symbol) in string.into_iter().enumerate() {
        let first_of_this_symbol = symbol_first(&outer_symbol, &grammar);
        for inner_symbol in &first_of_this_symbol {
            if inner_symbol != &EPSILON {
                // Add all the non-𝜖 symbols of first(outer_symbol) to first(string).
                first_set.insert(inner_symbol.clone());
            }
        }
        if first_of_this_symbol.contains(&EPSILON) {
            // If 𝜖 is in first(outer_symbol), also add the non-𝜖 symbols of
            // the first set of the next symbol in string.
            if idx == string_length - 1 {
                // If 𝜖 is in all symbols in the string, add 𝜖 to first(string).
                //
                // This is a horrible kludgy way to check whether or not this
                // is the last time through the loop.
                first_set.insert(EPSILON.clone());
            }
            continue;
        } else {
            // If 𝜖 is not in first(outer_symbol), do not keep adding symbols to first(string).
            break;
        }
    }
    first_set
}

/// Compute the closure of items.
///
/// This algorithm is from sec. 4.7.2 of the Dragon Book 2e, p. 261.
fn closure(items: HashSet<Item>, grammar: &Grammar) -> HashSet<Item> {
    let mut item_set: HashSet<Item> = HashSet::from(items);
    'repeat: loop {
        let old_item_set = item_set.clone();
        for item in item_set.clone() {
            for production in &grammar.productions[..] {
                if Symbol::NonTerminal(production.source) != item.production.result[item.dot] {
                    // We only want the productions that begin with the symbol after the dot.
                    continue;
                }
                // Get the string made up of the symbols immediately after the
                // symbol after the dot followed by the lookahead terminal.
                let mut little_item_set = Vec::from(&item.production.result[(item.dot + 1)..]);
                little_item_set.push(Symbol::Terminal(item.clone().lookahead));
                for terminal in string_first(little_item_set, grammar) {
                    item_set.insert(Item {
                        production: production.clone(),
                        dot: 0,
                        lookahead: terminal,
                    });
                }
            }
        }
        if item_set == old_item_set {
            // Repeat until no more items are added to item_set.
            break 'repeat;
        }
    }
    item_set
}

/// Compute the goto set for a given rule set.
///
/// Algorithm from Dragon Book 2e, sec. 4.7.2, p. 261.
fn goto(items: &HashSet<Item>, symbol: &Symbol, grammar: &Grammar) -> HashSet<Item> {
    // Initialize to the empty set.
    let mut result: HashSet<Item> = HashSet::new();
    for item in items {
        // Add all items the return set, advancing the dot by one.
        if item.production.result[item.dot] == *symbol {
            result.insert(Item {
                production: item.clone().production,
                dot: item.clone().dot + 1,
                lookahead: item.clone().lookahead,
            });
        }
    }

    return closure(result, grammar);
}

/// Compute the item set for the augmented grammar.
///
/// Algorithm from Dragon Book 2e, sec. 4.7.2, p. 261.
fn items(grammar: &Grammar) -> Vec<HashSet<Item>> {
    // The first entry in the grammar is the augmented start symbol.
    let mut items = Vec::from([closure(
        HashSet::from([Item {
            production: grammar.clone().start_production,
            dot: 0,
            lookahead: EOF.clone(),
        }]),
        &grammar,
    )]);

    'repeat_until_no_new_items: loop {
        let mut new_items = Vec::new();

        for item in &items {
            for symbol in &grammar.symbol_set {
                let goto_set = goto(item, symbol, &grammar);
                if !goto_set.is_empty() && !items.contains(&goto_set) {
                    new_items.push(goto_set);
                }
            }
        }

        if new_items.is_empty() {
            break 'repeat_until_no_new_items;
        }

        items.append(&mut new_items);
    }

    items
}

/// Construction the parsing tables from an augmented grammar.
///
/// Algorithm 4.56 from Dragon Book 2e, sec. 4.7.3, p. 265.
pub fn tables(grammar: Grammar) -> Result<(ActionTable, GotoTable), ()> {
    // FIXME: This is a horrible spaghetti mess that should surely be
    // refactored into something less unreadable.
    let item_sets: Vec<(usize, HashSet<Item>)> = items(&grammar).into_iter().enumerate().collect();
    let mut action_table: ActionTable = HashMap::new();
    let mut goto_table: GotoTable = HashMap::new();
    // Get our state ids from the order in which the item sets are generated.
    for (state_id, item_set) in &item_sets {
        for item in item_set {
            // If [A -> ɑ·aꞵ, b] is in item_set_i... then action_table[i, a] = shift(j),
            if item.dot < item.production.result.len() {
                match &item.production.result[item.dot] {
                    // ...where a is a terminal...
                    Symbol::Terminal(terminal) => {
                        // and goto(item_set_i, a) = item_set_j...
                        let goto_item_set =
                            goto(&item_set, &Symbol::Terminal(terminal.clone()), &grammar);
                        // The problem is to determine the state_id of goto_item_set.
                        for (candidate_state_id, candidate_item_set) in &item_sets {
                            // Do a simple linear search for the matching item
                            // set. This is going to be slow as all get out,
                            // but it only has to be run once at
                            // initialization, and besides, you know what they
                            // say about premature optimization...
                            // FIXME: Do this in a less carcinogenic way.
                            if &goto_item_set == candidate_item_set {
                                if action_table.contains_key(&(*state_id, terminal.clone())) {
                                    // If any conflicting actions result from the above rules, the
                                    // algorithm fails to produce a parser because the grammar is not
                                    // LR(1). FIXME: Do this error checking more cleanly.
                                    return Err(());
                                }
                                action_table.insert(
                                    (*state_id, terminal.clone()),
                                    Action::Shift(*candidate_state_id),
                                );
                            }
                        }
                    }
                    // The next symbol is not a terminal, so ignore it.
                    Symbol::NonTerminal(_) => {}
                };
            }
            // If [A -> ɑ·, a] is in item_set_i, A != grammar.start_symbol,
            // action_table[i, a] = reduce(A -> ɑ).
            if item.dot == item.production.result.len()
                && item.production.source != grammar.start_production.source
            {
                if action_table.contains_key(&(*state_id, item.lookahead.clone())) {
                    // If any conflicting actions result from the above rules, the
                    // algorithm fails to produce a parser because the grammar is not
                    // LR(1).
                    return Err(());
                }
                action_table.insert(
                    (*state_id, item.lookahead.clone()),
                    Action::Reduce(item.production.clone()),
                );
            }
            // If [S' -> S·, EOF] is in item_set_i, then set action_table[i, EOF] to accept.
            if item.production.source == grammar.start_production.source
                && item.dot == item.production.result.len()
                && item.lookahead == EOF
            {
                if action_table.contains_key(&(*state_id, EOF.clone())) {
                    // If any conflicting actions result from the above rules, the
                    // algorithm fails to produce a parser because the grammar is not
                    // LR(1).
                    return Err(());
                }
                action_table.insert((*state_id, EOF.clone()), Action::Accept);
            }
        }
        // The goto transitions for state state_id are constructed for all
        // nonterminals A using the goto function.
        for symbol in &grammar.symbol_set {
            match symbol {
                // Ignore terminals. FIXME: There must be a more idiomatic way to say this.
                Symbol::Terminal(_) => {}
                Symbol::NonTerminal(nonterminal) => {
                    let goto_item_set = goto(&item_set, &symbol, &grammar);
                    // The problem is to determine the state_id of goto_item_set.
                    for (candidate_state_id, candidate_item_set) in &item_sets {
                        // Do a simple linear search for the matching item
                        // set. This is going to be slow as all get out,
                        // but it only has to be run once at
                        // initialization, and besides, you know what they
                        // say about premature optimization...
                        if &goto_item_set == candidate_item_set {
                            goto_table.insert((*state_id, nonterminal), *candidate_state_id);
                        }
                    }
                }
            }
        }
    }
    Ok((action_table, goto_table))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (4.55) from the Dragon Book 2e, section 4.7.2, p. 263.
    fn example_grammar() -> Grammar {
        Grammar {
            symbol_set: vec![
                Symbol::NonTerminal("S'"),
                Symbol::NonTerminal("S"),
                Symbol::NonTerminal("C"),
                Symbol::Terminal(Terminal {
                    name: "c",
                    pattern: "c",
                    priority: 0,
                }),
                Symbol::Terminal(Terminal {
                    name: "d",
                    pattern: "d",
                    priority: 0,
                }),
            ],
            start_production: Production {
                source: "S'",
                result: vec![Symbol::NonTerminal("S")],
            },
            productions: vec![
                Production {
                    source: "S'",
                    result: vec![Symbol::NonTerminal("S")],
                },
                Production {
                    source: "S",
                    result: vec![Symbol::NonTerminal("C"), Symbol::NonTerminal("C")],
                },
                Production {
                    source: "C",
                    result: vec![
                        Symbol::Terminal(Terminal {
                            name: "c",
                            pattern: "c",
                            priority: 0,
                        }),
                        Symbol::NonTerminal("C"),
                    ],
                },
                Production {
                    source: "C",
                    result: vec![Symbol::Terminal(Terminal {
                        name: "d",
                        pattern: "d",
                        priority: 0,
                    })],
                },
            ],
        }
    }

    #[test]
    fn example_grammar_tables() {
        let grammar = example_grammar();
        let Ok((action_table, goto_table)) = tables(grammar) else {
            panic!()
        };
    }
}
