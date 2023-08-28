#include "../constants.glsl"

struct GetData {
    DATA_TYPE data;
    bool do_discard;
};

bool condition(int x, int y, int z, DATA_TYPE data);
DATA_TYPE get_identity();
DATA_TYPE accumulate(DATA_TYPE acc, DATA_TYPE data);

GetData get_data(int x, int y) {
    int to_z = get_z();

    GetData acc = GetData(get_identity(), true);

    for (int z = 0; z < to_z; z++) {
        if (!is_in_bounds(x, y, z)) {
            continue;
        }
        DATA_TYPE data = get_data_raw(x, y, z, TEXTURE_SIZE_X, TEXTURE_SIZE_Y);

#ifndef UNCONDITIONAL
        if (condition(x, y, z, data)) {
#endif
            acc.data = accumulate(acc.data, data);
            acc.do_discard = false;
#ifndef UNCONDITIONAL
        }
#endif
    }

    return acc;
}
