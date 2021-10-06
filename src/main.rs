use std::{str::FromStr, time::Instant};

use chess::{Board, Color, Piece};
use clap::{App, Arg};
use rand::prelude::SliceRandom;
use tch::{
    nn::{self, LinearConfig, Module, OptimizerConfig},
    Device, Tensor,
};

use serde::{Deserialize, Serialize};

const INPUTS: i64 = 768;
const MID_0: i64 = 512;
const OUT: i64 = 1;

const BATCH_SIZE: usize = 4096;
const PRINT_ITER: usize = 100;

const SCALE: f32 = 300.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct W {
    pub weights: Vec<Vec<Vec<f64>>>,
}

fn nnue(vs: &nn::Path) -> impl nn::Module {
    nn::seq()
        .add(nn::linear(
            vs / "input",
            INPUTS,
            MID_0,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add_fn(|x| x.clamp(0.0, 1.0))
        .add(nn::linear(
            vs / "out",
            MID_0,
            OUT,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add_fn(|x| x.sigmoid())
}

fn train(mut data: Vec<BoardEval>, wdl: bool, device: Device, out_file: &str) {
    let mut vs = nn::VarStore::new(device);
    vs.bfloat16();
    let net = nnue(&vs.root());
    let mut opt = nn::AdamW::default().build(&vs, 1e-3).unwrap();
    for epoch in 0.. {
        data.shuffle(&mut rand::thread_rng());
        let mut total_loss = Tensor::of_slice(&[0.0]).detach();
        let mut running_loss = Tensor::of_slice(&[0.0]).detach();
        let mut counter = 0;
        for index in (BATCH_SIZE..data.len()).step_by(BATCH_SIZE) {
            let slice = &data[index - BATCH_SIZE..index];
            let data = to_input_vectors(slice, wdl);
            let loss = net
                .forward(&data.inputs)
                .mse_loss(&data.outputs, tch::Reduction::Mean);
            opt.backward_step(&loss);

            for (_, value) in &mut vs.variables_.lock().unwrap().named_variables.iter_mut() {
                *value = value.clamp(-1.98, 1.98);
            }

            total_loss += &loss;
            running_loss += &loss;
            counter += 1;
            if counter % PRINT_ITER == (PRINT_ITER - 1) {
                println!(
                    "{} :{:?}",
                    counter,
                    (&running_loss / &Tensor::of_slice(&[PRINT_ITER as f32]).detach())
                );
                running_loss = Tensor::of_slice(&[0.0]).detach();
            }
        }
        let mut weights = vec![vec![]; 2];
        for (name, tensor) in &vs.variables() {
            let (index, input_size) = if name == "input.weight" {
                (0, INPUTS)
            } else if name == "out.weight" {
                (1, MID_0)
            } else {
                println!("WARNING: UNKNOWN LAYER");
                (1, 1)
            };
            weights[index] = tensor_to_slice(&tensor, input_size);
        }
        let weights = W { weights };
        let json = serde_json::to_string(&weights).unwrap();
        std::fs::write(out_file, json).unwrap();

        println!(
            "epoch {}: {:?}",
            epoch,
            total_loss / &Tensor::of_slice(&[counter as f32]).detach()
        );
    }
}

fn main() {
    let matches = App::new("NNUE Marlin")
        .version("v0.1-beta")
        .author("Doruk S. <dsekercioglu2003@gmail.com>")
        .about("Trains NNs for chess position evaluation")
        .arg(
            Arg::with_name("directory")
                .short("d")
                .long("dir")
                .value_name("DIR")
                .help("Loads positions from the files in this directory")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("out")
                .short("o")
                .long("out")
                .value_name("OUT")
                .help("Outputs the NNUE to this file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("wdl")
                .long("wdl")
                .help("Uses WDL info when training"),
        )
        .arg(
            Arg::with_name("cuda")
                .long("cuda")
                .help("Uses a CUDA GPU with the given id if available")
                .takes_value(true),
        )
        .get_matches();

    let files = matches.value_of("directory").unwrap();
    let output = matches.value_of("out").unwrap();
    let gpu = matches.value_of("cuda");
    let wdl = matches.value_of("wdl").is_some();

    let device = if let Some(gpu) = gpu {
        Device::Cuda(gpu.parse::<usize>().unwrap())
    } else {
        Device::Cpu
    };

    let time = Instant::now();
    let boards = parse_to_boards(files);

    println!("parsed {} boards in {:?}", boards.len(), time.elapsed());

    train(boards, wdl, device, output);
}

fn tensor_to_slice(tensor: &Tensor, input_size: i64) -> Vec<Vec<f64>> {
    Vec::<Vec<f64>>::from(tensor.detach().contiguous().view([-1, input_size]))
}

struct BoardEval {
    board: Board,
    eval: f32,
}

fn parse_to_boards(directory: &str) -> Vec<BoardEval> {
    let mut boards = vec![];

    let directory = std::fs::read_dir(directory).unwrap();
    for file in directory {
        let string_content = std::fs::read_to_string(file.unwrap().path()).unwrap();
        for line in string_content.lines() {
            let mut split = line.split("[");
            let fen = split.next().unwrap();
            let eval = split.next().unwrap();

            let board = Board::from_str(fen).unwrap();
            let eval = *&eval[0..eval.len() - 1].parse::<f32>().unwrap();
            if eval.abs() < 3000.0 {
                boards.push(BoardEval { board, eval });
            }
        }
    }
    boards
}

pub struct DataSet {
    inputs: Tensor,
    outputs: Tensor,
}

fn to_input_vectors(board_eval: &[BoardEval], wdl: bool) -> DataSet {
    let mut inputs = vec![];
    let mut outputs = vec![];
    for board_eval in board_eval {
        let (w, b) = to_input_vector(board_eval.board);
        inputs.extend(w.iter().cloned());
        inputs.extend(b.iter().cloned());
        let sigmoid_eval = if wdl {
            -board_eval.eval
        } else {
            1.0 / (1.0 + (-board_eval.eval as f32 / SCALE).exp())
        };
        outputs.push(sigmoid_eval);
        outputs.push(1.0 - sigmoid_eval);
    }
    DataSet {
        inputs: Tensor::of_slice(&inputs).view_(&[-1, INPUTS]).detach(),
        outputs: Tensor::of_slice(&outputs).view_(&[-1, 1]).detach(),
    }
}

fn to_input_vector(board: Board) -> ([f32; 768], [f32; 768]) {
    let mut w_perspective = [0_f32; 768];
    let mut b_perspective = [0_f32; 768];

    let white = *board.color_combined(Color::White);
    let black = *board.color_combined(Color::Black);

    let pawns = *board.pieces(Piece::Pawn);
    let knights = *board.pieces(Piece::Knight);
    let bishops = *board.pieces(Piece::Bishop);
    let rooks = *board.pieces(Piece::Rook);
    let queens = *board.pieces(Piece::Queen);
    let kings = *board.pieces(Piece::King);

    let array = [
        (white & pawns),
        (white & knights),
        (white & bishops),
        (white & rooks),
        (white & queens),
        (white & kings),
        (black & pawns),
        (black & knights),
        (black & bishops),
        (black & rooks),
        (black & queens),
        (black & kings),
    ];

    for (index, &pieces) in array.iter().enumerate() {
        for sq in pieces {
            let w_sq = sq.to_index();
            let b_sq = w_sq ^ 56;
            w_perspective[index * 64 + w_sq] = 1.0;
            b_perspective[((index + 6) % 12) * 64 + b_sq] = 1.0;
        }
    }

    (w_perspective, b_perspective)
}
