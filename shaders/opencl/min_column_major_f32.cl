__kernel void sum(
	__private ulong size_x,
	__private ulong size_y,
	__global float const* const data,
	__global float * const output
	) {
	const size_t local_x = get_global_id(0);
	
	float acc = INFINITY;
	
	for (size_t local_y = 0; local_y < size_y; local_y++) {
		acc = min(acc, data[local_y + (local_x * size_y)]);
	}

	output[local_x] = acc;
}
