//! Tokenizes stdin, one line of ids per input file: `ik-tok <checkpoint dir> [--no-special] < text`
use std::io::Read;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let ck = ik_engine::checkpoint::Checkpoint::open(std::path::Path::new(&args[1]))?;
    let t = std::time::Instant::now();
    let tok = ck.tokenizer()?;
    eprintln!("tokenizer loaded in {:?}", t.elapsed());
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
    let special = !args.iter().any(|a| a == "--no-special");
    let t = std::time::Instant::now();
    let ids = tok.encode(&text, special);
    eprintln!("{} tokens in {:?}", ids.len(), t.elapsed());
    println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
    assert_eq!(tok.decode(&ids, true), text, "decode does not round-trip");
    Ok(())
}
