//https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork

mod matrix2d;
mod nn;
pub use matrix2d::Matrix2D;
pub use nn::NeuralNetwork;

pub fn max_index(matrx: &Matrix2D) -> usize {
    let values = matrx.matrix();
    let mut max_val = &values[0];
    let mut max_index = 0usize;
    for i in 1..values.len() {
        if values[i] > *max_val {
            max_val = &values[i];
            max_index = i;
        }
    }
    max_index
}
