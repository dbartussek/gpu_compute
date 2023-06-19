fn main() {
    cc::Build::new()
        .file("src/an_external_function.c")
        .compile("an_external_function");
}
