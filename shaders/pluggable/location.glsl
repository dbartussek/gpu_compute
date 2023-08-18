
// How many layers of data are there?
int get_z();

// Which coordinate does this invocation operate on?
ivec2 get_coord();

#ifdef COMPUTE_SHADER

const uint WORKGROUP_SIZE = 64;

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(push_constant) uniform pc_l{ int z; } pc;
int get_z() {
    return pc.z;
}

ivec2 get_coord() {
    return ivec2(gl_GlobalInvocationID.xy);
}

#else

layout(location = 0) flat in uint instance_id;

int get_z() {
    return int(instance_id);
}

ivec2 get_coord() {
    return ivec2(gl_FragCoord.xy);
}

#endif
