use nalgebra::DMatrix;
use std::error;
use std::str::FromStr;
use std::{fs::File, u8};

// format is [label, pix-11, pix-12, pix-13, ...]
const TRAIN_CSV: &str = "data/mnist_train.csv";
const TEST_CSV: &str = "data/mnist_test.csv";

/// parses a csv for image data into a matrix
/// each row represents a single image where the first column is the label and the rest are the
/// pixel data. The number of columns should be 785 as the images are 28x28
fn parse_csv(file_path: &str) -> Result<DMatrix<u8>, Box<dyn error::Error>> {
    let file = File::open(file_path)?;
    let rdr = csv::Reader::from_reader(file);

    let mut data = Vec::new();
    let mut rows = 0;
    for result in rdr.into_records() {
        rows += 1;
        let record = result?;
        let record_iter: Result<Vec<_>, _> =
            record.into_iter().map(|x| u8::from_str(x.trim())).collect();
        let mut record_iter = record_iter?;
        data.append(&mut record_iter);
    }
    let cols = data.len() / rows;

    // rust-analyzer seems to be unhappy with this :(
    // related: https://github.com/rust-lang/rust-analyzer/issues/8654
    let matrix = DMatrix::from_row_slice(rows, cols, &data[..]);

    Ok(matrix)
}

fn main() {
    // We probably want to tranpose for the matrix operations
    let train_matrix = parse_csv(TRAIN_CSV).expect("failed to parse training csv");
    let test_matrix = parse_csv(TEST_CSV).expect("failed to parse testing csv");

    assert!(train_matrix.ncols() == 785);
    assert!(test_matrix.ncols() == 785);
}
