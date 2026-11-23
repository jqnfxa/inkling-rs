#![allow(long_running_const_eval)]
//! Prints the PTX module: `ik-ptx > kernels.ptx`
fn main() {
    print!("{}", ik_kernels::gpu::PTX);
}
