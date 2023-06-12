use crate::args::Args;
use crate::check_error;
use crate::kernel::Kernel;
use cuda_runtime_sys::{
    cudaError, cudaLaunchKernel, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize,
    cudaStream_t, dim3,
};
use std::ptr;

pub struct Stream {
    pointer: cudaStream_t,
}

impl Stream {
    pub fn new() -> Result<Self, cudaError> {
        let mut pointer: cudaStream_t = ptr::null_mut();
        unsafe { check_error(cudaStreamCreate(&mut pointer as *mut cudaStream_t))? }
        Ok(Self { pointer })
    }

    pub fn launch(
        &mut self,
        kernel: &Kernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &Args,
        shared_mem: usize,
    ) -> Result<(), cudaError> {
        let grid_dim = dim3 {
            x: grid_dim.0,
            y: grid_dim.1,
            z: grid_dim.2,
        };
        let block_dim = dim3 {
            x: block_dim.0,
            y: block_dim.1,
            z: block_dim.2,
        };
        unsafe {
            check_error(cudaLaunchKernel(
                kernel.function,
                grid_dim,
                block_dim,
                args.as_args(),
                shared_mem,
                self.pointer,
            ))
        }
    }

    pub fn wait(&self) -> Result<(), cudaError> {
        unsafe { check_error(cudaStreamSynchronize(self.pointer)) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            check_error(cudaStreamDestroy(self.pointer)).unwrap();
        }
    }
}
