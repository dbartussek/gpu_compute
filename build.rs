fn main() {
    cc::Build::new()
        .file("src/an_external_function.c")
        .compile("an_external_function");

    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .cuda(true)
            .file("src/cuda/cuda_sum.cu")
            .compile("cuda_accumulate");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rerun-if-changed=src/cuda/cuda_sum.cu");
    }
}
