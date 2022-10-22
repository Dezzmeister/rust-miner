#[cfg(feature = "kernels")]
fn build_kernels() {
    cuda_builder::CudaBuilder::new("../miner-kernel")
        .copy_to("kernels/miner.ptx")
        .build()
        .unwrap();
}

fn main() {
    #[cfg(feature = "kernels")]
    build_kernels();
}