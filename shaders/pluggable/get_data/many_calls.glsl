#include "../constants.glsl"

struct GetData {
    uint data;
    bool do_discard;
};

bool condition(int x, int y, int z, uint data);

GetData get_data(int x, int y) {
    int z = int(instance_id);
    uint data = get_data_raw(x, y, z);

    if (!condition(x, y, z, data)) {
        return GetData(0, true);
    }

    return GetData(data, false);
}
