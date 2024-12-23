use burn::backend::{wgpu::WgpuDevice, Wgpu};

// Type alias for the backend to use.
// type MyBackend = Wgpu;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    tensor_basics::initialization::initializate::<MyBackend>(device.clone());
    tensor_basics::ownership::ownership::<MyBackend>(device.clone());
}