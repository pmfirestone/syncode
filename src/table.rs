// src/table.rs
//! Produce action and goto tables for the use of Syncode in the [`crate::parser`] module.
//!
//! Based on the relevant sections of the Dragon Book, 2e.
//!
//! This whole module is a spaghetti mess for which I'm going to go to
//! hell. I'm sure that someone who understood the relevant algorithms better
//! than I do could restate them in a way that produced better code, but I'm
//! translating them to Rust directly from the book as I understand it. There
//! is work to be done here in cleaning the code and improving its efficiency;
//! right now the goal is functioning code, nothing more, nothing less.

use std::cell::LazyCell;
use std::collections::{HashMap, HashSet};

use crate::types::*;

/// A convenience terminal representing the empty string.
const EPSILON: LazyCell<Terminal<'static>> = LazyCell::new(|| Terminal::new("epsilon", "", 0));

/// A convenience terminal representing the end of the input.
fn eof() -> Terminal<'static> {
    Terminal::new("$", "", 0)
}

/// Construct the first set of a given symbol.
///
/// The algorithm comes from sec. 4.4.2 of the Dragon Book 2e, p. 221.
fn symbol_first(symbol: &Symbol, grammar: &Grammar) -> HashSet<Terminal<'static>> {
    match symbol {
        // If symbol is a terminal, then first(symbol) = {symbol}.
        Symbol::Terminal(terminal) => HashSet::from([terminal.clone()]),
        // If symbol is a nonterminal...
        Symbol::NonTerminal(nonterminal) => {
            let mut first_set = HashSet::new();
            for production in grammar.productions.clone() {
                // And symbol -> y1y2...yk for some k >= 1,
                if production.lhs == *nonterminal {
                    for symbol in &production.rhs {
                        // Place the contents of the first set of the resulting
                        // symbol into this symbol's first set...
                        let first = symbol_first(symbol, grammar);
                        first_set.extend(first.clone().into_iter());
                        if production.rhs == vec![Symbol::Terminal(EPSILON.clone())] {
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
        }
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
        let first_of_this_symbol = symbol_first(&outer_symbol, grammar);
        for inner_symbol in &first_of_this_symbol {
            if inner_symbol != &*EPSILON {
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
    let mut item_set: HashSet<Item> = items;
    'repeat: loop {
        let old_item_set = item_set.clone();
        for item in item_set.clone() {
            for production in &grammar.productions[..] {
                if item.dot == item.production.rhs.len() {
                    // We only want productions that don't have the dot at the
                    // end (and to avoid a panic when indexing at the next
                    // check).
                    continue;
                }
                if Symbol::NonTerminal(production.lhs) != item.production.rhs[item.dot] {
                    // We only want the productions that begin with the symbol after the dot.
                    continue;
                }
                // Get the string made up of the symbols immediately after the
                // symbol after the dot followed by the lookahead terminal.
                let mut little_item_set = Vec::from(&item.production.rhs[(item.dot + 1)..]);
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
        // We only want items where the dot is not at the end of the result yet.
        if item.dot == item.production.rhs.len() {
            continue;
        }
        // Add all items the return set, advancing the dot by one.
        if item.production.rhs[item.dot] == *symbol {
            result.insert(Item {
                production: item.clone().production,
                dot: item.clone().dot + 1,
                lookahead: item.clone().lookahead,
            });
        }
    }

    closure(result, grammar)
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
            lookahead: eof(),
        }]),
        grammar,
    )]);

    'repeat_until_no_new_items: loop {
        let mut new_items = Vec::new();

        for item in &items {
            for symbol in &grammar.symbol_set {
                let goto_set = goto(item, symbol, grammar);
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

/// Build the action table for the grammar.
fn action_table(grammar: &Grammar) -> ActionTable {
    let mut action_table: ActionTable = HashMap::new();
    let item_sets: Vec<HashSet<Item>> = items(&grammar);
    for (state_id, item_set) in item_sets.clone().into_iter().enumerate() {
        for item in &item_set {
            // If [A -> ɑ·aꞵ, b] is in item_set_i...
            if item.dot < item.production.rhs.len() {
                match &item.production.rhs[item.dot] {
                    // ...where a is a terminal...
                    Symbol::Terminal(terminal) => {
                        // and goto(item_set_i, a) = item_set_j...
                        let goto_item_set =
                            goto(&item_set, &Symbol::Terminal(terminal.clone()), &grammar);
                        let Ok(goto_state_id) = find_state_id(&goto_item_set, &item_sets) else {
                            panic!(
                                "When adding a shift action for terminal {:#?}, failed to find the state id for goto item set {:#?}",
                                terminal, goto_item_set
                            );
                        };
                        checked_insert(
                            state_id,
                            terminal.clone(),
                            Action::Shift(goto_state_id),
                            &mut action_table,
                        );
                    }
                    // The next symbol is not a terminal, so ignore it.
                    Symbol::NonTerminal(_) => {}
                };
            }
            // If [A -> ɑ·, a] is in item_set_i, A != grammar.start_symbol,
            // action_table[i, a] = reduce(A -> ɑ).
            if item.dot == item.production.rhs.len()
                && item.production.lhs != grammar.start_production.lhs
            {
                checked_insert(
                    state_id,
                    item.lookahead.clone(),
                    Action::Reduce(item.production.clone()),
                    &mut action_table,
                );
            }
            // If [S' -> S·, EOF] is in item_set_i, then set action_table[i, EOF] to accept.
            if item.production.lhs == grammar.start_production.lhs
                && item.dot == item.production.rhs.len()
                && item.lookahead == eof()
            {
                checked_insert(state_id, eof(), Action::Accept, &mut action_table);
            }
        }
    }
    action_table
}

/// Construct the goto table for this grammar.
fn goto_table(grammar: &Grammar) -> GotoTable {
    let mut goto_table: GotoTable = HashMap::new();
    let item_sets: Vec<HashSet<Item>> = items(&grammar);
    for (state_id, item_set) in item_sets.clone().into_iter().enumerate() {
        // The goto transitions for state state_id are constructed for all
        // nonterminals A using the goto function.
        for symbol in &grammar.symbol_set {
            match symbol {
                // Ignore terminals. FIXME: There must be a more idiomatic way to say this.
                Symbol::Terminal(_) => {}
                Symbol::NonTerminal(nonterminal) => {
                    let goto_item_set = goto(&item_set, &symbol, &grammar);
                    let Ok(goto_state_id) = find_state_id(&goto_item_set, &item_sets) else {
                        // Sometimes there just isn't a goto for a given (item_set, symbol) pair.
                        continue;
                    };
                    goto_table.insert((state_id, *nonterminal), goto_state_id);
                }
            }
        }
    }
    goto_table
}

/// Construct the parsing tables from an augmented grammar.
///
/// Algorithm 4.56 from Dragon Book 2e, sec. 4.7.3, p. 265.
pub fn tables(grammar: Grammar) -> (ActionTable, GotoTable) {
    // FIXME: This is a horrible spaghetti mess that should surely be
    // refactored into something less unreadable.
    (action_table(&grammar), goto_table(&grammar))
}

/// Find the state_id of this item.
///
/// Assume that the state_ids are the order in which the states were added to
/// the item_sets vec. Return Err(()) if the item set wasn't found.
fn find_state_id(item_set: &HashSet<Item>, item_sets: &[HashSet<Item>]) -> Result<usize, ()> {
    for (candidate_state_id, candidate_item_set) in item_sets.iter().enumerate() {
        if candidate_item_set == item_set {
            return Ok(candidate_state_id);
        }
    }
    Err(())
}

/// Try to insert a rule into the action table, failing if there's already a rule there.
///
/// Failure here indicates that the grammar isn't LR.
fn checked_insert(
    state_id: usize,
    terminal: Terminal<'static>,
    action: Action,
    table: &mut ActionTable,
) {
    if table.contains_key(&(state_id, terminal.clone())) {
        // If any conflicting actions result from the above rules, the
        // algorithm fails to produce a parser because the grammar is not
        // LR(1). FIXME: Do this error checking more cleanly.
        panic!(
            "Whil inserting an action into the action table, discovered a conflicting action:\nstate_id: {state_id}, terminal: {:#?}, existing action: {:#?}, new action: {:#?}",
            terminal.clone(),
            table.get(&(state_id, terminal)),
            action
        );
    }
    // then action_table[i, a] = shift(j).
    table.insert((state_id, terminal.clone()), action);
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
                Symbol::Terminal(Terminal::new("c", "c", 0)),
                Symbol::Terminal(Terminal::new("d", "d", 0)),
            ],
            start_production: Production {
                lhs: "S'",
                rhs: vec![Symbol::NonTerminal("S")],
            },
            productions: vec![
                Production {
                    lhs: "S'",
                    rhs: vec![Symbol::NonTerminal("S")],
                },
                Production {
                    lhs: "S",
                    rhs: vec![Symbol::NonTerminal("C"), Symbol::NonTerminal("C")],
                },
                Production {
                    lhs: "C",
                    rhs: vec![
                        Symbol::Terminal(Terminal::new("c", "c", 0)),
                        Symbol::NonTerminal("C"),
                    ],
                },
                Production {
                    lhs: "C",
                    rhs: vec![Symbol::Terminal(Terminal::new("d", "d", 0))],
                },
            ],
        }
    }

    #[test]
    fn example_grammar_tables() {
        let grammar = example_grammar();
        let (action_table, goto_table) = tables(grammar);
        eprintln!("{:#?}", action_table);
        eprintln!("{:#?}", goto_table);
    }
}
