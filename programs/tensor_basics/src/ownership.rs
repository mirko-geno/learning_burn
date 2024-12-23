/*
Most burn's Tensor operations tak ownership of variables
so in order to reuse them clone method must be used.
Clone doesn't copy all data, but add one to a reference
system similar to Rust RC.
*/

use burn::{
    tensor::Tensor,
    prelude::*
};


pub fn ownership<B:Backend>(device: B::Device) {
    let input = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let min = input.clone().min();
    let max = input.clone().max();
    let input = (input.clone() - min.clone()).div(max - min);
    println!("{}", input);
}