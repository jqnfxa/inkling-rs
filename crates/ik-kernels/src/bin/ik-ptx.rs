#![allow(long_running_const_eval)]
//! Prints the PTX modules (gpu, then attn): `ik-ptx > kernels.ptx`
fn main() {
    print!("{}{}", ik_kernels::gpu::PTX, ik_kernels::attn::PTX);
}
