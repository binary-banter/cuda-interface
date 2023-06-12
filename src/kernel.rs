use std::os::raw::c_void;

/// `Kernel` holds the kernel function.
pub struct Kernel {
    pub(crate) function: *const c_void,
}

impl Kernel {
    /// Create new `Kernel` from extern C.
    pub fn new(function: unsafe extern "C" fn() -> c_void) -> Self {
        Self {
            function: function as *const c_void,
        }
    }
}
