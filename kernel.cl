__kernel void matrixMull(__global int *A, __global int *B, __global int *C)												{												
	int idx_row = get_global_id(0);				
	int idx_col = get_global_id(1);				
	int col_size = get_global_size(1);			
	int address = idx_row*col_size + idx_col;
	int address2 = idx_row + col_size*idx_col;

	for (int i = 0; i < col_size; i++){

		C[address] = A[address] * B[address2];
		address++;
		address2++;
	}
				
	return;										
};