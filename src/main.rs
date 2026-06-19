use std::env;

mod llm;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <words...>", args[0]);
        return;
    } else if args.len() == 2 && args[1] == "setup" {
        eprintln!("Error: Too many arguments. Maximum allowed is 10.");
        return;
    }

    let args = &args[1..];
    let words = args.join(" ");

    println!("{}", words);
}
