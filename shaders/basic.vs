#version 460

layout(location = 0) out vec2 tex_coord;
layout(location = 1) flat out uint instance_id;

// When drawing a clipped triangle, set scale to 2 to fill the entire screen
layout(constant_id = 0) const int DATA_SCALE = 1;
// A single full screen quad when drawn as triangle strip
const vec2 DATA[4] = {vec2(0, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1)};

void main() {
    vec2 position = DATA[gl_VertexIndex] * DATA_SCALE;

    gl_Position = vec4((position - 0.5) * 2, 0.0, 1.0);
    tex_coord = position;
    instance_id = gl_InstanceIndex;
}
