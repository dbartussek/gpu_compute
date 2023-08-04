#include "../constants.glsl"

struct GetData {
    DATA_TYPE data;
    bool do_discard;
};

bool condition(int x, int y, int z, DATA_TYPE data);

GetData get_data(int x, int y) {
    int z = get_z();
    DATA_TYPE data = get_data_raw(x, y, z);

#ifndef UNCONDITIONAL
    if (!condition(x, y, z, data)) {
        return GetData(0, true);
    }
#endif

    return GetData(data, false);
}
