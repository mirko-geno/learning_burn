use burn::{
    nn::{
        // Dropout, DropoutConfig, 
        Linear, LinearConfig,
        Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig}
    },
    prelude::*
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    pool: MaxPool2d,
    // dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu
}

impl<B: Backend> Model<B> {
    /// # Shapes
    /// - Images [batch_size, height, width]
    /// - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims(); // [batch_size, height=28, with=28]
        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 16, 26, 26]
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 32, 24, 24]
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x); // [batch_size, 64, 22, 22]
        // let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // [batch_size, _, 21, 21]

        let x = self.conv4.forward(x); // [batch_size, 64, 19, 19]
        let x = self.activation.forward(x);
        let x = self.conv5.forward(x); // [batch_size, 128, 17, 17]
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // [batch_size, _, 16, 16]

        let x = self.conv6.forward(x); // [batch_size, 256, 14, 14]
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // [batch_size, _, 13, 13]

        let x = x.reshape([batch_size, 256 * 13 * 13]);

        let x = self.linear1.forward(x);
        // let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
        
    }
}


#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    // #[config(default = "0.5")]
    // dropout: f64
}

impl ModelConfig {
    // Returns the initialized model
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 16], [3, 3]).init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
            conv3: Conv2dConfig::new([32, 64], [3, 3]).init(device),

            conv4: Conv2dConfig::new([64, 128], [3, 3]).init(device),
            conv5: Conv2dConfig::new([128, 192], [3, 3]).init(device),
            conv6: Conv2dConfig::new([192, 256], [3, 3]).init(device),

            pool: MaxPool2dConfig::new([2, 2]).init(),

            // dropout: DropoutConfig::new(self.dropout).init(),

            linear1: LinearConfig::new(256 * 13 * 13, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),

            activation: Relu::new()
        }
    }
}