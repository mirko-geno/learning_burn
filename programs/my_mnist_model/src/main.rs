use my_mnist_model::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::Autodiff,
    // backend::{wgpu::WgpuDevice, Wgpu},
    optim::AdamConfig,
    data::dataloader::Dataset,
};
use burn_cuda::{Cuda, CudaDevice};


fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = CudaDevice::default();
    let artifact_dir = "/tmp/model";

    let model = my_mnist_model::model::ModelConfig::new(10, 512).init::<MyBackend>(&device);
    println!("Model: {}", model);
    drop(model);

    my_mnist_model::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    my_mnist_model::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
        );
    
}