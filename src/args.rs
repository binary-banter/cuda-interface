use crate::buffer::Buffer;
use std::marker::PhantomData;
use std::os::raw::c_void;

/// Arguments to pass to a kernel.
#[derive(Default)]
pub struct Args<'a> {
    /// Pointers to the arguments.
    args: Vec<*mut c_void>,

    // PhantomData is used to guarantee the args field outlives the underlying arguments.
    phantom: PhantomData<&'a ()>,
}

impl<'a> Args<'a> {
    /// Add a new argument to `Args`.
    pub fn add_arg(&mut self, arg: impl ToArg<'a>) {
        self.args.push(arg.to_arg());
    }

    /// Convert `Args` into pointer that points to its `args` field.
    pub(crate) fn as_args(&self) -> *mut *mut c_void {
        self.args.as_ptr() as *mut *mut c_void
    }
}

/// This trait allows types to be added to `Args`.
pub trait ToArg<'a> {
    /// Turns `self` into pointer.
    fn to_arg(self) -> *mut c_void;
}

impl<'a, T> ToArg<'a> for &'a mut Buffer<T> {
    fn to_arg(self) -> *mut c_void {
        (&mut self.pointer) as *mut *mut c_void as *mut c_void
    }
}

impl<'a> ToArg<'a> for &'a u32 {
    fn to_arg(self) -> *mut c_void {
        self as *const u32 as *mut u32 as *mut c_void
    }
}

impl<'a> ToArg<'a> for &'a f32 {
    fn to_arg(self) -> *mut c_void {
        self as *const f32 as *mut f32 as *mut c_void
    }
}

// todo: use a macro to autogenerate a bunch of standard types (e.g. u8, u16, etc.)
