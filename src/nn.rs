use super::matrix2d::Matrix2D;
use anyhow::Result;
use bincode::{config, Decode, Encode};
use std::fs::File;
use std::io::{Read, Write};

// 神经网络
#[derive(Encode, Decode)]
pub struct NeuralNetwork {
    inodes: usize, //输入节点数
    hnodes: usize, //隐藏节点数
    onodes: usize, //输出节点数
    lr: f32,       //学习率

    // 连接权重矩阵, wih (W input_hidden) 和 who (W hidden_output)
    // 数组中的权重是 w_i_j, 链接是从节点i到下一层的节点j
    // w11  w21
    // w12  w22 等
    wih: Matrix2D,
    who: Matrix2D,
}

impl NeuralNetwork {
    //初始化神经网络
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f32,
    ) -> NeuralNetwork {
        NeuralNetwork {
            inodes: input_nodes,
            hnodes: hidden_nodes,
            onodes: output_nodes,
            lr: learning_rate,
            wih: Matrix2D::random(hidden_nodes, input_nodes),
            who: Matrix2D::random(output_nodes, hidden_nodes),
        }
    }

    //训练神经网络
    pub fn train(&mut self, inputs_list: &[f32], targets_list: &[f32]) {
        //1维数组 转换成2维数组
        let inputs = Matrix2D::from(inputs_list);
        let targets = Matrix2D::from(targets_list);

        //计算进入隐藏层的信号
        let hidden_inputs = Matrix2D::dot(&self.wih, &inputs);
        //计算从隐藏层产生的信号
        let hidden_outputs = Matrix2D::sigmoid(&hidden_inputs);

        //计算进入最终输出层的信号
        let final_inputs = Matrix2D::dot(&self.who, &hidden_outputs);
        //计算输出层产生的信号
        let final_outputs = Matrix2D::sigmoid(&final_inputs);

        //错误 = 目标-实际输出
        let output_errors = targets - final_outputs.clone();

        //隐藏层的错误
        //error_hidden = W(T)_hidden_output ● error_output
        let hidden_errors = Matrix2D::dot(&Matrix2D::transpose(&self.who), &output_errors);

        //我们需要调整每层的权重。
        //"隐藏层->输出层"的权重使用output_errors(来调整)
        //"输入层->隐藏层"的权重使用hidden_errors(来调整)

        //更新隐藏层到输出层之间连接的权重
        self.who += self.lr
            * Matrix2D::dot(
                &(output_errors * final_outputs.clone() * (1.0 - final_outputs)),
                &Matrix2D::transpose(&hidden_outputs),
            );

        //更新输入层和隐藏层之间连接的权重
        self.wih += self.lr
            * Matrix2D::dot(
                &(hidden_errors * hidden_outputs.clone() * (1.0 - hidden_outputs)),
                &Matrix2D::transpose(&inputs),
            );
    }

    //查询神经网络
    pub fn query(&self, inputs_list: &[f32]) -> Matrix2D {
        //1维数组 转换成2维数组
        let inputs = Matrix2D::from(inputs_list);
        //计算进入隐藏层的信号
        let hidden_inputs = Matrix2D::dot(&self.wih, &inputs);
        //计算从隐藏层产生的信号
        let hidden_outputs = Matrix2D::sigmoid(&hidden_inputs);

        //计算进入最终输出层的信号
        let final_inputs = Matrix2D::dot(&self.who, &hidden_outputs);
        //计算输出层产生的信号
        let final_outputs = Matrix2D::sigmoid(&final_inputs);

        final_outputs
    }

    pub fn put_weights(&mut self, weights: &[f32]) {
        let mut i = 0;
        for row in self.wih.matrix_mut() {
            for x in row {
                *x = weights[i];
                i += 1;
            }
        }
        for row in self.who.matrix_mut() {
            for x in row {
                *x = weights[i];
                i += 1;
            }
        }
    }

    //返回网络所需的权重总数
    pub fn get_number_of_weights(&self) -> usize {
        let w1 = self.wih.matrix().iter().flatten();
        let w2 = self.who.matrix().iter().flatten();
        w1.count() + w2.count()
    }

    pub fn get_weights(&self) -> Vec<f32> {
        let mut w1: Vec<f32> = self.wih.matrix().clone().into_iter().flatten().collect();
        let w2: Vec<f32> = self.who.matrix().clone().into_iter().flatten().collect();
        w1.extend_from_slice(&w2);
        w1
    }

    /// 存储到文件
    pub fn save(&self, file_name: &str) -> Result<()> {
        let mut file = File::create(file_name)?;
        file.write_all(self.serialize()?.as_slice())?;
        Ok(())
    }

    /// 读入文件
    pub fn load(file_name: &str) -> Result<NeuralNetwork> {
        let mut data_file = File::open(file_name)?;
        let mut data = vec![];
        data_file.read_to_end(&mut data)?;
        Ok(Self::deserialize(&data)?)
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        let bin = bincode::encode_to_vec(self, config::standard())?;
        Ok(bin)
    }

    pub fn deserialize(encoded: &[u8]) -> Result<NeuralNetwork> {
        Ok(bincode::decode_from_slice(encoded, config::standard())?.0)
    }
}
