# nnue_marlin

NNUE Marlin is used to create Neural Networks for chess position evaluation.

It currently supports different configurations via changing code manually however the command line arguments will be expanded upon later...


In order to get started, generate txt files with the following format:

<fen_0> <evaluation_0>\
<fen_1> <evaluation_1>\
...

Put the files you wish to use for training in a single directory and then call it via
`cargo run --release -- --dir <dir> --out <nnue_file>.json`
or `cargo build --release` and then call it via `target/release/nnue_marlin --dir <dir> --out <nnue_file>.json`

The tch-rs crate used to train the neural networks doesn't support M1 Macs yet, so it'll require `--target=x86_64-apple-darwin` while building in order to run the program on Rosetta-2.