#include "../constants.glsl"

struct GetData {
    DATA_TYPE data;
    bool do_discard;
};

bool condition(int x, int y, int z, DATA_TYPE data);
DATA_TYPE get_identity();
DATA_TYPE accumulate(DATA_TYPE acc, DATA_TYPE data);

GetData get_data(int x, int y) {
    int toZ = get_z();

    DATA_TYPE acc = get_identity();

    for (int z = 0; z < toZ; z++) {
        DATA_TYPE data = get_data_raw(x, y, z);

#ifndef UNCONDITIONAL
        if (condition(x, y, z, data)) {
#endif
            acc = accumulate(acc, data);
#ifndef UNCONDITIONAL
        }
#endif
    }

    return GetData(acc, false);
}
