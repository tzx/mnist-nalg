use nalgebra::{DMatrix, DVector, SMatrix, SVector};

const NUMBERS: usize = 10;
pub const NUM_DATAPOINTS: usize = 784;

/// holds the data for a dataset
/// labels is a column vector of the rows
/// data is a matrix where each column represents a single datapoint normalized between 0 and 1
pub struct DataSet {
    labels: DVector<u8>,
    data: DMatrix<f64>,
    num_data_points: usize,
}

pub fn build_dataset(matrix: &DMatrix<f64>) -> DataSet {
    let label_slice = matrix.column(0);
    let data_slice = matrix.columns(1, matrix.ncols() - 1);

    let labels = DVector::from_iterator(
        label_slice.nrows(),
        label_slice.map(|x| x as u8).into_iter().cloned(),
    );
    let data = DMatrix::from_iterator(
        data_slice.nrows(),
        data_slice.ncols(),
        data_slice.iter().cloned(),
    )
    .transpose()
        / 255.0;
    let num_data_points = data.nrows();

    DataSet {
        labels,
        data,
        num_data_points,
    }
}

impl DataSet {
    pub fn get_num_data_points(&self) -> usize {
        self.num_data_points
    }
}

pub struct Params<const N: usize> {
    w1: SMatrix<f64, N, NUMBERS>,
    b1: SVector<f64, NUMBERS>,
    w2: SMatrix<f64, NUMBERS, NUMBERS>,
    b2: SVector<f64, NUMBERS>,
}

impl<const N: usize> std::fmt::Debug for Params<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "W1: {:?}, b1: {:?}, W2: {:?}, b2: {:?}",
            self.w1, self.b1, self.w2, self.b2
        )
    }
}

pub fn init_params<const N: usize>() -> Params<N> {
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Uniform::<f64>::new(0.0, 1.0);

    let w1 = SMatrix::<f64, N, NUMBERS>::from_distribution(&dist, &mut rng);
    let b1 = SVector::<f64, NUMBERS>::from_distribution(&dist, &mut rng);
    let w2 = SMatrix::<f64, NUMBERS, NUMBERS>::from_distribution(&dist, &mut rng);
    let b2 = SVector::<f64, NUMBERS>::from_distribution(&dist, &mut rng);

    return Params { w1, b1, w2, b2 };
}
