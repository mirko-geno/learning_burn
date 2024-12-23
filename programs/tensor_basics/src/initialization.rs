use burn::{
    tensor::Tensor,
    prelude::*
};


pub fn initializate<B: Backend>(device: B::Device) {
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Tensors are defined by their dimensionality
    let tensor_1 = Tensor::<B, 1>::from_floats(floats, &device);
    // Creation of a second tensor filled with ones, with the same shape as the first
    let _tensor_2 = Tensor::<B, 1>::ones_like(&tensor_1);

    
    // Initialization from a given Backend (Wgpu)
    // let _wgpu_tensor = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0], &device);

    // Initialization from a generic Backend
    let _generic_tensor = Tensor::<B, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);


    // Initialization using from_floats (Recommended for f32 ElementType)
    // Will be converted to TensorData internally.
    let _tensor = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);

    
    // Initialization of Int Tensor from array slices
    let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let _tensor = Tensor::<B, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);


    // Initialization from a custom type
    struct BodyMetrics {
        age: i8,
        height: i16,
        weight: f32
    }

    let bmi = BodyMetrics{
        age: 25,
        height: 180,
        weight: 80.0
    };

    let data = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
    let _tensor = Tensor::<B, 1>::from_data(data, &device);
}