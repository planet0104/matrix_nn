use bincode::{Decode, Encode};
use rand::Rng;
use std::f32::consts::E;
use std::ops::{AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Encode, Decode, Debug, Clone)]
pub struct Matrix2D {
    matrix: Vec<Vec<f32>>,
}

fn log(n: f32) -> f32 {
    n.log(E)
}

fn logit(p: f32) -> f32 {
    log(p / (1.0 - p))
}

//Matrix2D-Matrix2D
impl Sub for Matrix2D {
    type Output = Matrix2D;

    fn sub(self, other: Matrix2D) -> Matrix2D {
        //矩阵相减
        let mut matrix: Vec<Vec<f32>> = vec![];
        //每一行
        for row in 0..self.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            //每一列
            for col in 0..self.matrix[row].len() {
                new_row.push(self.matrix[row][col] - other.matrix[row][col]);
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }
}

//Matrix2D*Matrix2D
impl Mul for Matrix2D {
    type Output = Matrix2D;
    fn mul(self, other: Matrix2D) -> Matrix2D {
        //矩阵相乘
        let mut matrix: Vec<Vec<f32>> = vec![];
        //每一行
        for row in 0..self.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            //每一列
            for col in 0..self.matrix[row].len() {
                new_row.push(self.matrix[row][col] * other.matrix[row][col]);
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }
}

// +=
impl AddAssign for Matrix2D {
    fn add_assign(&mut self, other: Matrix2D) {
        //矩阵加上一个矩阵
        //每一行
        for row in 0..self.matrix.len() {
            //每一列
            for col in 0..self.matrix[row].len() {
                self.matrix[row][col] += other.matrix[row][col];
            }
        }
    }
}

// += f32
impl AddAssign<f32> for Matrix2D {
    fn add_assign(&mut self, other: f32) {
        //矩阵加上一个矩阵
        //每一行
        for row in 0..self.matrix.len() {
            //每一列
            for col in 0..self.matrix[row].len() {
                self.matrix[row][col] += other;
            }
        }
    }
}

// -= f32
impl SubAssign<f32> for Matrix2D {
    fn sub_assign(&mut self, other: f32) {
        //矩阵减去一个矩阵
        //每一行
        for row in 0..self.matrix.len() {
            //每一列
            for col in 0..self.matrix[row].len() {
                self.matrix[row][col] -= other;
            }
        }
    }
}

// /= f32
impl DivAssign<f32> for Matrix2D {
    fn div_assign(&mut self, other: f32) {
        //矩阵减去一个矩阵
        //每一行
        for row in 0..self.matrix.len() {
            //每一列
            for col in 0..self.matrix[row].len() {
                self.matrix[row][col] /= other;
            }
        }
    }
}

// *= f32
impl MulAssign<f32> for Matrix2D {
    fn mul_assign(&mut self, other: f32) {
        //矩阵减去一个矩阵
        //每一行
        for row in 0..self.matrix.len() {
            //每一列
            for col in 0..self.matrix[row].len() {
                self.matrix[row][col] *= other;
            }
        }
    }
}

//f32*Matrix2D
impl Mul<Matrix2D> for f32 {
    type Output = Matrix2D;

    fn mul(self, rhs: Matrix2D) -> Matrix2D {
        //矩阵乘以一个数
        let mut matrix: Vec<Vec<f32>> = vec![];
        //每一行
        for row in 0..rhs.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            //每一列
            for col in 0..rhs.matrix[row].len() {
                new_row.push(rhs.matrix[row][col] * self);
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }
}

//f32-Matrix2D
impl Sub<Matrix2D> for f32 {
    type Output = Matrix2D;
    fn sub(self, rhs: Matrix2D) -> Matrix2D {
        //一个数减去矩阵
        let mut matrix: Vec<Vec<f32>> = vec![];
        //每一行
        for row in 0..rhs.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            //每一列
            for col in 0..rhs.matrix[row].len() {
                new_row.push(self - rhs.matrix[row][col]);
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }
}

impl Matrix2D {
    pub fn random(rows: usize, columns: usize) -> Matrix2D {
        //产生一个随机的矩阵
        let mut matrix: Vec<Vec<f32>> = vec![];
        for _ in 0..rows {
            let mut row: Vec<f32> = vec![];
            for _ in 0..columns {
                row.push(rand::thread_rng().gen_range(-0.5..0.51));
            }
            matrix.push(row);
        }
        Matrix2D { matrix: matrix }
    }

    pub fn new(matrix: Vec<Vec<f32>>) -> Matrix2D {
        Matrix2D { matrix: matrix }
    }

    pub fn matrix(&self) -> &Vec<Vec<f32>> {
        &self.matrix
    }

    pub fn matrix_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.matrix
    }

    pub fn min(&self) -> f32 {
        let a: Vec<&f32> = self
            .matrix
            .iter()
            .map(|row| row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .collect();
        **(a.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
    }

    pub fn max(&self) -> f32 {
        let a: Vec<&f32> = self
            .matrix
            .iter()
            .map(|row| row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .collect();
        **(a.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
    }

    pub fn dot_old(m1: &Matrix2D, m2: &Matrix2D) -> Matrix2D {
        //矩阵点乘
        let mut matrix: Vec<Vec<f32>> = vec![];

        //m1的每一行
        for row in 0..m1.matrix.len() {
            let mut new_row: Vec<f32> = vec![];

            //m2的每一个列
            for col_2 in 0..m2.matrix[row].len() {
                let mut sum = 0f32;
                //m1当前行的
                for col in 0..m1.matrix[row].len() {
                    //m2取col列对应的行, col_2对应m2每一个列
                    sum += m1.matrix[row][col] * m2.matrix[col][col_2];
                }
                new_row.push(sum);
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }

    pub fn dot(ma: &Matrix2D, mb: &Matrix2D) -> Matrix2D {
        let row_a = ma.matrix.len();
        let row_b = mb.matrix.len();
        let column_a = ma.matrix[0].len();
        let column_b = mb.matrix[0].len();
        if column_a != row_b {
            panic!("矩阵无法相乘!");
        } else {
            let mut mc: Vec<Vec<f32>> = vec![vec![0f32; column_b]; row_a];
            for i in 0..row_a {
                for j in 0..column_b {
                    for k in 0..column_a {
                        mc[i][j] += ma.matrix[i][k] * mb.matrix[k][j];
                    }
                }
            }
            Matrix2D { matrix: mc }
        }
    }

    //一维数组转换二维矩阵
    pub fn from(values: &[f32]) -> Matrix2D {
        Matrix2D {
            matrix: values.iter().map(|x| vec![*x]).collect(),
        }
    }

    //S型函数
    pub fn sigmoid(inputs: &Matrix2D) -> Matrix2D {
        //S形响应曲线
        //当已知神经细胞所有输入x权重的乘积之和时，这一方法将它送入S形的激励函数
        //pub const ACTIVATION_RESPONSE: f32 = 1.5;
        let mut matrix: Vec<Vec<f32>> = vec![];
        for row in 0..inputs.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            for col in 0..inputs.matrix[row].len() {
                //expit(x) = 1/(1+exp(-x))
                new_row.push(1.0 / (1.0 + (-inputs.matrix[row][col]).exp()));
                //new_row.push(1.0/(1.0+(-inputs.matrix[row][col]/1.5).exp()));
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }

    //反向S型函数
    pub fn inverse_sigmoid(inputs: &Matrix2D) -> Matrix2D {
        //S形响应曲线
        //当已知神经细胞所有输入x权重的乘积之和时，这一方法将它送入S形的激励函数
        let mut matrix: Vec<Vec<f32>> = vec![];
        for row in 0..inputs.matrix.len() {
            let mut new_row: Vec<f32> = vec![];
            for col in 0..inputs.matrix[row].len() {
                //logit(p) = log(p/(1-p))
                new_row.push(logit(inputs.matrix[row][col]));
            }
            matrix.push(new_row);
        }
        Matrix2D { matrix: matrix }
    }

    /** 转置矩阵 */
    pub fn transpose(m: &Matrix2D) -> Matrix2D {
        let mut matrix: Vec<Vec<f32>> = vec![];

        for i in 0..m.matrix[0].len() {
            let mut new_row: Vec<f32> = vec![];
            for j in 0..m.matrix.len() {
                new_row.push(m.matrix[j][i]);
            }
            matrix.push(new_row);
        }

        Matrix2D { matrix: matrix }
    }
}
