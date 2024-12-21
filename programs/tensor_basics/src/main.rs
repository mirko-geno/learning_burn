use burn::{
    tensor::Tensor,
    backend::Wgpu
};

// Type alias for the backend to use.
type Backend = Wgpu;

fn main() {
    // Default::default() inherits its type from context
    let device = Default::default();

    // Creation of two tensors, the first with explicit values and the second
    // one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors
    // Using clone method is not recommended, used only for the example
    println!("Sum:{}", tensor_1.clone() + tensor_2.clone());
    println!("Sub: {}", tensor_1.clone() - tensor_2.clone());
}