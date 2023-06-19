#version 460
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coord;
layout(location = 1) flat out uint instance_id;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coord = (position + 1) / 2;
    instance_id = gl_InstanceIndex;
}
