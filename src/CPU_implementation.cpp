
//////////////////////////////////////////////////////////////////////////////
// Lossless compression using Exponential Golomb coding
// Created : 01/02/2016
//////////////////////////////////////////////////////////////////////////////

// Library list
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>
#include <CT/DataFiles.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

using namespace HDF5;

// The data cube has 24576 elements each with 6*6*6 points inside
const int side = 6;
const int side2 = side * side; //side squared
const int side3 = side2 * side;//side cubed
const int num_element = 24576;

// Define desired Q-factor for compression
const float Qfactor = 1000.0;
const int bits_of_int = sizeof(int) * 8;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
/*
 * Reads the value and ensures lower boundaries are not violated.
 */
double val(const std::vector<float>& h_input, int elem, int x, int y, int z){
	return ((x<0)||(y<0)||(z<0)) ? 0.0 : h_input[elem + x * side2 + y * side + z];
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
 * Returns the absolute value of an integer.
 * Remark: Don't use it with the largest negative integer!
 */
int iabs(int v){
	return (v >= 0)? v : -v;
}

/*
 * Casts the double value into the nearest integer.
 * Remark: Don't use it with values that won't fit integers!
 */
int iround(float v){
	return (v >= 0.0)? v + 0.5 : v - 0.5;
}

/*
 * It performs:
 * a. Lorenzo prediction for estimating the value of the scalar field at each sample from the values at processed neighbours.
 * b. Exponential Golomb coding for generating the 64-bit codeword.
 */
void calculateHost (std::vector<float>& h_input, std::vector<unsigned int>& h_output, int & compressed_size) {
	int x, y, z;
	double v;
	double lorenzo;
	int Qres;
	unsigned long cw;		//CodeWord
	int lcw;				//CodeWord Length
	int B = 0;				//Address of current write position
	int b = bits_of_int;	//Number of free bits available in the current write position
	h_output[0] = 0;
	for (int el = 0; el < h_input.size(); el += side3)
		for (x = 0; x < side; x++)
			for (y = 0; y < side; y++)
				for (z = 0; z < side; z++){
					v = val(h_input, el, x, y, z);
					// Lorenzo prediction algorithm
					lorenzo = (val(h_input, el, x - 1, y - 1, z - 1)
							- val(h_input, el, x - 1, y - 1, z))
							+ (val(h_input, el, x - 1, y, z)
							- val(h_input, el, x - 1, y, z - 1))
							+ (val(h_input, el, x, y - 1, z)
							- val(h_input, el, x, y - 1, z - 1))
							+ val(h_input, el, x, y, z - 1);
					Qres = iround((v - lorenzo) / Qfactor);
					h_input[el + x * side2 + y * side + z] = lorenzo + Qres * Qfactor;

					//(Exp-Golomb(n) = Elias-gamma(n+1)) + [sign /*if(n!=0)*/]
					if(Qres){
						cw = (iabs(Qres) + 1)<<1;
						lcw = (ilog2(cw)<<1); //len(cw-s) + len(cw'-s) + 1
						if(Qres<0){
							cw |= 1; //sets the minus sign
						}
					} else { //Qres is 0 so cw = '1'
						cw = 1; 
						lcw = 1;
					}

					b -= lcw;
					while(b < 0){
						h_output[B++] |= cw >> -b;
						b += bits_of_int;
						h_output[B] = 0;
					}
					h_output[B] |= cw << b;
				}
	compressed_size = (B + 1) * bits_of_int - b;
}

//////////////////////////////////////////////////////////////////////////////
// Open an existing HDF5 file.
//////////////////////////////////////////////////////////////////////////////

void readData(std::vector<double>& buffer_p1){
	hid_t file_id = H5Fopen(<your-input-data-file-in-HDF5-format>, H5F_ACC_RDONLY, H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, "/DG_Solution", H5P_DEFAULT); // For viewing the dataset, refer to http://vitables.org/

	double* input_data;

	input_data = (double*) malloc (num_element * side3 * 5 * sizeof (double));

	herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_data);
	int param = 4;
	int index = 0;

	for (int element=0; element < num_element; element++) {
		for (int z=0; z < side; z++) {
			for (int y=0; y < side; y++) {
				for (int x=0; x < side; x++) {
					buffer_p1[index] = input_data[(((element*side+x)*side+y)*side+z)*5+param];
					index = index + 1;
				}
			}
		}
	}
	free(input_data);

	H5Dclose(dataset_id);
	H5Fclose(file_id);
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	cl::Context context(CL_DEVICE_TYPE_GPU);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Declare some values
	std::size_t wgSize = 96; // Number of work items per work group
	std::size_t count = num_element * side3; // Overall number of work items = Number of elements
	std::size_t size = count * sizeof (double); // Size of data in bytes
	std::size_t sizeo = count * sizeof (int); // Size of data output in bytes
	std::size_t sizel = num_element * sizeof (int); // Size of length buffers in bytes

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/GPU_implementation.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	//OpenCL::buildProgram(program, devices);
	OpenCL::buildProgram(program, devices, "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));

	// Create a kernel object
	cl::Kernel compressBlockKernel(program, "compressBlockKernel");
	cl::Kernel copyHighBitsKernel(program, "copyHighBitsKernel");
	cl::Kernel copyLowBitsKernel(program, "copyLowBitsKernel");
	cl::Kernel prefixSumKernel (program, "prefixSumKernel");
	cl::Kernel blockAddKernel (program, "blockAddKernel");

	std::cout << "sizeof(double) = " << sizeof(double) << std::endl;
	std::cout << "sizeof(int) = " << sizeof(int) << std::endl;

	// Allocate space for input data and for output data from CPU and GPU on the host
	std::vector<double> h_input (count);
	std::vector<float> h_inputf (count);
	std::vector<unsigned int> h_outputCpu (count);
	std::vector<unsigned int> h_outputGpu (count);
	std::vector<unsigned int> h_inter (num_element*320);  //intermediate compression results (for each full element)
	std::vector<unsigned int> h_length (num_element);
	std::vector<unsigned int> h_pos (num_element);
	std::vector<unsigned int> h_temp (num_element/wgSize);
	std::vector<unsigned int> h_temp2 (num_element/wgSize/wgSize);

	// Allocate space for input and output data on the device
	cl::Buffer d_input (context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output (context, CL_MEM_READ_WRITE, sizeo);
	cl::Buffer d_inter (context, CL_MEM_READ_WRITE, num_element*320*sizeof(int));
	cl::Buffer d_length (context, CL_MEM_READ_WRITE, sizel);
	cl::Buffer d_pos (context, CL_MEM_READ_WRITE, sizel);
	cl::Buffer d_temp (context, CL_MEM_READ_WRITE, sizel/wgSize);
	cl::Buffer d_temp2 (context, CL_MEM_READ_WRITE, sizel/wgSize/wgSize);
	cl::Buffer d_temp3 (context, CL_MEM_READ_WRITE, sizel/wgSize/wgSize/wgSize*0 + wgSize*4);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, sizeo);
	memset(h_outputGpu.data(), 255, sizeo);
	memset(h_inter.data(), 255, num_element * 320 * sizeof(int));
	memset(h_length.data(), 255, sizel);
	memset(h_pos.data(), 255, sizel);
	memset(h_temp.data(), 255, sizel/wgSize);
	memset(h_temp.data(), 255, sizel/wgSize/wgSize);
	queue.enqueueWriteBuffer(d_input, true, 0, size/2, h_inputf.data());
	queue.enqueueWriteBuffer(d_output, true, 0, sizeo, h_outputGpu.data());
	queue.enqueueWriteBuffer(d_inter, true, 0, num_element*320*sizeof(int), h_inter.data());
	queue.enqueueWriteBuffer(d_length, true, 0, sizel, h_length.data());
	queue.enqueueWriteBuffer(d_pos, true, 0, sizel, h_length.data());
	queue.enqueueWriteBuffer(d_temp, true, 0, sizel/wgSize, h_temp.data());
	queue.enqueueWriteBuffer(d_temp2, true, 0, sizel/wgSize/wgSize, h_temp2.data());

	// Initialize input data with values from the file
	readData(h_input);
	for(int s = 0; s< count; s++)
		h_inputf[s] = h_input[s];

	// Copy input data to device
	cl::Event copy1;
	queue.enqueueWriteBuffer(d_input, true, 0, size/2, h_inputf.data(), NULL, &copy1);

	// Do calculation on the host side
	int compressed_size = 0;
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	calculateHost(h_inputf, h_outputCpu, compressed_size);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	std::cout << "Original size:  " << count * sizeof(double) * 8 << " bits" << std::endl;
	std::cout << "Compressed size: " << compressed_size << " bits" << std::endl;
	std::cout << "Compression ratio: " << count * sizeof(double) * 8.0 / compressed_size << std::endl;

	// Launch kernel on the device
	// Do the compression for each element
	cl::Event exec1;
	compressBlockKernel.setArg<cl::Buffer>(0, d_input);
	compressBlockKernel.setArg<cl::Buffer>(1, d_inter);
	compressBlockKernel.setArg<cl::Buffer>(2, d_length);
	compressBlockKernel.setArg<cl_int>(3, side);
	compressBlockKernel.setArg<cl_float>(4, Qfactor);
	queue.enqueueNDRangeKernel(compressBlockKernel, 0, num_element, wgSize, NULL, &exec1);

	// Add the bit lengths into bit positions
	cl::Event exec2;
	prefixSumKernel.setArg<cl::Buffer>(0, d_length);
	prefixSumKernel.setArg<cl::Buffer>(1, d_pos);
	prefixSumKernel.setArg<cl::Buffer>(2, d_temp);
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, num_element, wgSize, NULL, &exec2);

	cl::Event exec2b;
	prefixSumKernel.setArg<cl::Buffer>(0, d_temp);
	prefixSumKernel.setArg<cl::Buffer>(1, d_temp);
	prefixSumKernel.setArg<cl::Buffer>(2, d_temp2);
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, num_element/wgSize, wgSize, NULL, &exec2b);

	cl::Event exec2c;
	prefixSumKernel.setArg<cl::Buffer>(0, d_temp2);
	prefixSumKernel.setArg<cl::Buffer>(1, d_temp2);
	prefixSumKernel.setArg<cl::Buffer>(2, d_temp3);
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, num_element/wgSize/wgSize, wgSize, NULL, &exec2c);

	cl::Event exec2d;
	prefixSumKernel.setArg<cl::Buffer>(0, d_temp3);
	prefixSumKernel.setArg<cl::Buffer>(1, d_temp3);
	prefixSumKernel.setArg<cl::Buffer>(2, cl::Buffer ());
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, num_element/wgSize/wgSize/wgSize*0+wgSize, wgSize, NULL, &exec2d);

	cl::Event exec3c;
	blockAddKernel.setArg<cl::Buffer>(0, d_temp2);
	blockAddKernel.setArg<cl::Buffer>(1, d_temp3);
	queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, num_element/wgSize/wgSize, wgSize, NULL, &exec3c);

	cl::Event exec3;
	blockAddKernel.setArg<cl::Buffer>(0, d_temp);
	blockAddKernel.setArg<cl::Buffer>(1, d_temp2);
	queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, num_element/wgSize, wgSize, NULL, &exec3);

	cl::Event exec3b;
	blockAddKernel.setArg<cl::Buffer>(0, d_pos);
	blockAddKernel.setArg<cl::Buffer>(1, d_temp);
	queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, num_element, wgSize, NULL, &exec3b);

	// Write the high parts of the shifted compression string
	cl::Event exec4;
	copyHighBitsKernel.setArg<cl::Buffer>(0, d_inter);
	copyHighBitsKernel.setArg<cl::Buffer>(1, d_output);
	copyHighBitsKernel.setArg<cl::Buffer>(2, d_pos);
	copyHighBitsKernel.setArg<cl_int>(3, side3);
	queue.enqueueNDRangeKernel(copyHighBitsKernel, 0, num_element, wgSize, NULL, &exec4);

	// Write the low parts of the shifted compression string
	cl::Event exec4b;
	copyLowBitsKernel.setArg<cl::Buffer>(0, d_inter);
	copyLowBitsKernel.setArg<cl::Buffer>(1, d_output);
	copyLowBitsKernel.setArg<cl::Buffer>(2, d_pos);
	copyLowBitsKernel.setArg<cl_int>(3, side3);
	queue.enqueueNDRangeKernel(copyLowBitsKernel, 0, num_element, wgSize, NULL, &exec4b);

	// Copy output data back to host
	cl::Event copy2;
	queue.enqueueReadBuffer(d_output, true, 0, sizeo, h_outputGpu.data(), NULL, &copy2);

	queue.enqueueReadBuffer(d_pos, true, 0, sizel, h_pos.data());
	std::cout << "GPU Compressed size: " << h_pos[num_element-1] << " bits" << std::endl;

	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime1 = OpenCL::getElapsedTime(exec1);
	Core::TimeSpan gpuTime2 = OpenCL::getElapsedTime(exec2);
	Core::TimeSpan gpuTime2b = OpenCL::getElapsedTime(exec2b);
	Core::TimeSpan gpuTime2c = OpenCL::getElapsedTime(exec2c);
	Core::TimeSpan gpuTime3 = OpenCL::getElapsedTime(exec3);
	Core::TimeSpan gpuTime3b = OpenCL::getElapsedTime(exec3b);
	Core::TimeSpan gpuTimeAdd = gpuTime2  + gpuTime2b + gpuTime2c + gpuTime3  + gpuTime3b;
	Core::TimeSpan gpuTime4 = OpenCL::getElapsedTime(exec4);
	Core::TimeSpan gpuTime4b = OpenCL::getElapsedTime(exec4b);
	Core::TimeSpan gpuTime = gpuTime1 + gpuTimeAdd + gpuTime4 + gpuTime4b;
	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
	Core::TimeSpan copyTime = copyTime1 + copyTime2;
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;
	std::cout << "CPU Time:\t\t\t" << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time:\t\t" << copyTime.toString() << std::endl;
	std::cout << "GPU Time compress:\t\t" << gpuTime1.toString() << std::endl;
	std::cout << "GPU Time add:\t\t\t" << gpuTimeAdd.toString() << std::endl;
	std::cout << "GPU Time shift high part:\t" << gpuTime4.toString() << std::endl;
	std::cout << "GPU Time shift low part:\t" << gpuTime4b.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy:\t" << gpuTime.toString() << " (speed-up = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "GPU Time with memory copy:\t" << overallGpuTime.toString() << " (speed-up = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;

	// Check whether results are correct
	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < count; i++) {
		// Allow small differences between CPU and GPU results (due to different rounding behaviour)
		if (!(std::abs (h_outputCpu[i] - h_outputGpu[i]) <= 1e-5)) {
			if (errorCount < 15)
				std::cout << "Result for " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ((h_outputGpu[i]>h_outputCpu[i])? ">": "<") << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
