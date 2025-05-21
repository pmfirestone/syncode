use syncode_core::mask::dfa_mask_store;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("Qwen/Qwen3-235B-A22B", None)?;
    //    let mask_store = dfa_mask_store();
    Ok(())
}
