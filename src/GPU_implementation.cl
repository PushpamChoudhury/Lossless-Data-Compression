
#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//General idea:
// 1 Compress each element in parallel
// 2 sum the sizes of the results (PrefixSum)
// 3 copy the first parts
// 4 copy the last parts
// Done


__constant const int bits_of_int = sizeof(int) * 8;

/*
 * Reads the value and ensures lower boundaries are not violated.
 */
double val(__local float* d_input, int elem, int x, int y, int z, int side, int side2, int side3){
	return ((x<0)||(y<0)||(z<0)) ? 0.0 : d_input[elem + x * side2 + y * side + z];
}

/*
 * Returns the bit address of the MSB whose value is 1, or 0 if all bits are 0.
 */
int ilog2(unsigned long w){
	int l = 0;
	while (w>>=1){
		l++;
	}
	return l;
}

/*
 * Returns the same as ilog2 for w != 0. Reads the result from the binary representation of float.
 */
char clog2f(float* w){
	return (*((unsigned int*) w) >> 23) - 127;
}

/*
 * Returns the absolute value of an integer.
 * Remark: Don't use it with the largest negative integer!
 */
int iabs(int v){
	return (v >= 0)? v : -v;
}

/*
 * Casts the double into the nearest integer.
 * Remark: Don't use it with values that won't fit integers!
 */
int iround(float v){
	return (v >= 0.0f)? v + 0.5f : v - 0.5f;
}

__kernel void compressBlockKernel (__global float* d_input, __global unsigned int* d_output, __global int* compressed_size, int side, float Qfactor) {
	int index = get_global_id(0);
	int side2 = side * side;
	int side3 = side2 * side;
	int side3l = 76; //dark magic
	int lel = get_local_id(0) * side3l;	//First address of the current element's data
	int gel = index * 320;//side3;	//First address of the current element's data

	int x=1, y, z;

	__local float aux_input[WG_SIZE * 76];

	float v;
	float lorenzo;
	int Qres;
	unsigned long cw;		//CodeWord
	int lcw;				//CodeWord Length
	int B = gel;				//Address of current write position
	int b = bits_of_int;	//Number of free bits available in the current write position
	d_output[B] = 0;
	for (z = 0; z < side2; z++)
		aux_input[lel + z] = 0;//*/
	for (int xx = 0; xx < side; xx++){
		for (y = 0; y < side; y++)
			for (z = 0; z < side; z++){
				v = d_input[index*side3 + xx * side2 + y * side + z];
				lorenzo = (val(aux_input, lel, x - 1, y - 1, z - 1, side, side2, side3l)
						- val(aux_input, lel, x - 1, y - 1, z, side, side2, side3l))
						+ (val(aux_input, lel, x - 1, y, z, side, side2, side3l)
						- val(aux_input, lel, x - 1, y, z - 1, side, side2, side3l))
						+ (val(aux_input, lel, x, y - 1, z, side, side2, side3l)
						- val(aux_input, lel, x, y - 1, z - 1, side, side2, side3l))
						+ val(aux_input, lel, x, y, z - 1, side, side2, side3l);
				Qres = iround((v - lorenzo) / Qfactor);
				aux_input[lel + x * side2 + y * side + z] = lorenzo + Qres * Qfactor;

				//(Exp-Golomb(n) = Elias-gamma(n+1)) + [sign if(n!=0)]
				if(Qres){
					cw = (iabs(Qres) + 1)<<1;
					float cwv = cw;
					lcw = clog2f(&cwv)*2;
					if(Qres<0){
						cw |= 1; //sets the minus sign
					}

				}else{ //Qres is 0 so cw = '1'
					cw = 1;
					lcw = 1;
				}

				b -= lcw;
				while(b < 0){
					d_output[B++] |= cw >> -b;
					b += bits_of_int;
					d_output[B] = 0; //may we assume clean memory?
				}
				d_output[B] |= cw << b;
			}
		x = 1;
		for (z = 0; z < side2; z++)
			aux_input[lel + z] = aux_input[lel + side2 + z];
	}
	compressed_size[index] = (B + 1 - gel) * bits_of_int - b;
}

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void prefixSumKernel(__global int* d_input, __global int* d_output, __global int* temp) {
	int i = get_global_id(0);
	int li  = get_local_id(0);
	int g = get_group_id(0);
	int s;

	__local int aux[WG_SIZE];

	aux[li] = d_input[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int k = 1; k < WG_SIZE; k <<= 1){
		s = aux[li];
		if(i >= k){
			s += aux[li - k];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		aux[li] = s;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	d_output[i] = aux[li];

	if(temp && (li == WG_SIZE - 1)){
		temp[g] = aux[li];
	}
}

__kernel void blockAddKernel(__global int* d_inout, __global int* temp) {
	int i = get_global_id(0);
	int g = get_group_id(0);

	if(g != 0){
		d_inout[i] += temp[g - 1];
	}
}

__kernel void copyHighBitsKernel (__global unsigned int* d_input, __global unsigned int* d_output, __global int* d_pos, int side3) {
	int index = get_global_id(0);
	int el = index * 320;//side3;
	int el_1 = el + 1;

	int p;
	unsigned int cw, cwn;

	int shift = 0, firstB = 0, firstBsize;
	if(index){
		shift = (d_pos[index-1] & (bits_of_int-1));
		firstB = (d_pos[index-1] >> ilog2(bits_of_int)) + ((shift)? 1: 0); //Start from the next completely free address
	}

	firstBsize = ((d_pos[index]) >> ilog2(bits_of_int)) - firstB;
	int endbit = d_pos[index] & (bits_of_int-1);

	if(endbit){ //Check if we have high bits to set at the end
		firstBsize++;
	}
	int b = bits_of_int - shift;
	//Writes new highest parts of output

	/*for(p = 0; p<firstBsize; p++){
		cw = d_input[el + p];
		d_output[firstB + p] = cw << (bits_of_int - shift);
	}*/
	cw = d_input[el];
	for(p = 0; p<firstBsize-1; p++){
		cwn = d_input[el_1 + p];
		d_output[firstB + p] = (cw << b) | ((shift) ? (cwn >> shift) : 0);
		cw = cwn;
	}
	if(firstBsize){
		cwn = d_input[el_1 + p];
		d_output[firstB + p] = (cw << b) | ((shift&&((endbit>shift)||(endbit == 0))) ? (cwn >> shift) : 0);
	}
}

__kernel void copyLowBitsKernel (__global unsigned int* d_input, __global unsigned int* d_output, __global int* d_pos, int side3) {
	int index = get_global_id(0);
	int el = index * 320;//side3;

	int p = 0;
	unsigned int cw;

	int B = 0, shift = 0;//, Bsize = 0;
	if(index){
		shift = (d_pos[index-1] & (bits_of_int-1));
		B = d_pos[index-1] >> ilog2(bits_of_int);
	}

	if(shift){
		cw = d_input[el + p];
		d_output[B + p] |= cw >> shift;
	}
}
