#include "../constants.glsl"

struct GetData {
    uint data;
    bool do_discard;
};

bool condition(int x, int y, int z, uint data);
uint get_identity();
uint accumulate(uint acc, uint data);

GetData get_data(int x, int y) {
    int toZ = get_z();

    uint acc = get_identity();

    for (int z = 0; z < toZ; z++) {
        uint data = get_data_raw(x, y, z);
        if (condition(x, y, z, data)) {
            acc = accumulate(acc, data);
        }
    }

    return GetData(acc, false);
}
