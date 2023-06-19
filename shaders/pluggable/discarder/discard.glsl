
GetData get_data_discarder(int x, int y) {
    GetData d = get_data(x, y);
    if (d.do_discard) {
        discard;
    }

    return d;
}
