mod record_data;

use std::error;
use std::{fs::File, u8};
use std::str::FromStr;
use nalgebra::{DMatrix};

// format is [label, pix-11, pix-12, pix-13, ...]
const TRAIN_CSV: &str = "data/mnist_train.csv";

fn parse_csv() -> Result<(), Box<dyn error::Error>> {
    let file = File::open(TRAIN_CSV)?;
    let rdr = csv::Reader::from_reader(file);

    let mut data = Vec::new();

    let mut rows = 0;
    for result in rdr.into_records() {
        rows += 1;
        let record = result?;
        let record_iter: Result<Vec<_>, _> = record.into_iter().map(|x| u8::from_str(x.trim())).collect();
        let mut record_iter = record_iter?;
        data.append(&mut record_iter);
    }
    let cols = data.len() / rows;

    Ok(())
}

fn main() {
    parse_csv();
}
