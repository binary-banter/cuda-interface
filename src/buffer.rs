use crate::check_error;
use cuda_runtime_sys::{cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset};
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::os::raw::c_void;
use std::ptr;

/// Buffer holds a device pointer to `length` number of `T`.
pub struct Buffer<T> {
    pub(crate) pointer: *mut c_void,
    length: usize,
    phantom: PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(length: usize) -> Result<Self, cudaError> {
        let mut pointer: *mut c_void = ptr::null_mut();
        unsafe {
            check_error(cudaMalloc(&mut pointer as *mut _, length * size_of::<T>()))?;
        }

        Ok(Self {
            pointer,
            length,
            phantom: PhantomData::default(),
        })
    }

    /// reads data length of bytes from buffer using an offset
    /// panics if it offset + data length overflows the buffer size
    pub fn read(&self, offset: usize, data: &mut [T]) -> Result<(), cudaError> {
        assert!(offset + data.len() <= self.length);

        unsafe {
            check_error(cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.pointer.add(offset * size_of::<T>()),
                size_of_val(data),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))?;
        }

        Ok(())
    }

    pub fn read_all(&self) -> Result<Vec<T>, cudaError>
    where
        T: Default + Clone,
    {
        let mut vec = vec![T::default(); self.length];
        self.read(0, &mut vec)?;
        Ok(vec)
    }

    /// writes data length of bytes to buffer using an offset
    /// panics if it offset + data length overflows the buffer size
    pub fn write(&mut self, offset: usize, data: &[T]) -> Result<(), cudaError> {
        assert!(offset + data.len() <= self.length);

        unsafe {
            check_error(cudaMemcpy(
                self.pointer.add(offset * size_of::<T>()),
                data.as_ptr() as *const c_void,
                size_of_val(data),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }

        Ok(())
    }

    pub fn zero(&mut self) -> Result<(), cudaError> {
        unsafe { check_error(cudaMemset(self.pointer, 0, self.length * size_of::<T>())) }
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            check_error(cudaFree(self.pointer)).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;

    #[test]
    fn write_read_u8() {
        let mut buffer = Buffer::<u8>::new(1).unwrap();
        buffer.write(0, &[0x12]).unwrap();
        let mut b = 0;
        buffer.read(0, slice::from_mut(&mut b)).unwrap();
        assert_eq!(b, 0x12);
    }

    #[test]
    fn write_read_u32() {
        let mut buffer = Buffer::<u32>::new(1).unwrap();
        buffer.write(0, &[0x1111_2222]).unwrap();
        let mut b = 0;
        buffer.read(0, slice::from_mut(&mut b)).unwrap();
        assert_eq!(b, 0x1111_2222);
    }
}
