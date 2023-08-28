#include "../constants.glsl"

struct GetData {
    DATA_TYPE data;
    bool do_discard;
};

bool condition(int x, int y, int z, DATA_TYPE data);

GetData get_data(int x, int y) {
    int z = get_z();

    if (!is_in_bounds(x, y, z)) {
        return GetData(0, true);
    }

    DATA_TYPE data = get_data_raw(x, y, z, TEXTURE_SIZE_X, TEXTURE_SIZE_Y);

#ifndef UNCONDITIONAL
    if (!condition(x, y, z, data)) {
        return GetData(0, true);
    }
#endif

    return GetData(data, false);
}
