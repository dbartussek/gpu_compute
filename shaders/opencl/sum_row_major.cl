__kernel void sum(
	__private ulong size_x,
	__private ulong size_y,
	__global uint const* const data,
	__global uint * const output
	) {
	const size_t local_x = get_global_id(0);
	
	uint acc = 0;
	
	for (size_t local_y = 0; local_y < size_y; local_y++) {
		acc += data[(local_y * size_x) + local_x];
	}

	output[local_x] = acc;
}
