
int an_external_function() {
    return 0;
}

int do_virtual_call(int (*arg)()) {
    return arg();
}
