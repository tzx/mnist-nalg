use nalgebra::{DVector, DMatrix};

/// holds the data for a dataset
/// labels is a column vector of the rows
/// data is a matrix where each column represents a single datapoint
pub struct DataSet {
    labels: DVector<u8>,
    data: DMatrix<u8>,
}

pub fn build_dataset(matrix: &DMatrix<u8>) -> DataSet {
    let label_slice = matrix.column(0);
    let data_slice = matrix.columns(1, matrix.ncols() - 1);

    let labels = DVector::from_iterator(label_slice.nrows(), label_slice.iter().cloned());
    let data = DMatrix::from_iterator(data_slice.nrows(), data_slice.ncols(), data_slice.iter().cloned()).transpose();

    DataSet {
        labels,
        data,
    }
}
