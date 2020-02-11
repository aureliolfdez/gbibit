/*
 ============================================================================
 Name        : BiBit.cu
 Author      : Aurelio Lopez-Fernandez
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <thread>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>
#include <iterator>
#include <utility>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <set>
#include <vector>
#include <map>
#include <unordered_set>
#include <mutex>
using namespace std;

namespace {
template<typename T>
std::size_t make_hash(const T& v) {
	return std::hash<T>()(v);
}

void hash_combine(std::size_t& h, const std::size_t& v) {
	h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
}

template<typename T>
struct hash_container {
	size_t operator()(const T& v) const {
		size_t h = 0;
		for (const auto& e : v) {
			hash_combine(h, make_hash(e));
		}
		return h;
	}
};
}

namespace std {
template<typename T, typename U>
struct hash<pair<T, U>> {
	size_t operator()(const pair<T, U>& v) const {
		size_t h = make_hash(v.first);
		hash_combine(h, make_hash(v.second));
		return h;
	}
};

template<typename ... T>
struct hash<vector<T...>> : hash_container<vector<T...>> {
};

template<typename ... T>
struct hash<map<T...>> : hash_container<map<T...>> {
};
}

char *inputFile; // Input matrix.
ulong cCols, cColsTotal, cRows, cRowsPerThread, patternSize; // Pattern size: We will work with 32 bit or 64 bit in the encoding phase.
int cMnr, cMnc; // minimum number of rows allowed in a valid bicluster and minimum number of columns allowed in a valid bicluster
int output, deviceCount;
__constant__ ulong rowsPerThread; // 8 bytes
__constant__ ulong cols; // 8 bytes
__constant__ ulong numThreads; // 8 bytes
__constant__ ulong rows; // 8 bytes
__constant__ int mnr; // 4 bytes
__constant__ int mnc; // 4 bytes
__device__ unsigned long long int totalBiclusters;
__device__ unsigned long long int numPatFiltered;
ulong maxPatterns;
ulong maxThreadsPerBlock;
ulong maxBlocksPerGrid;
ulong maxIteratorGPU;
ulong lastBlocksGrid;
long long totales;
std::unordered_set<std::vector<uint64_t>> setPatterns64;
std::unordered_set<std::vector<uint32_t>> setPatterns32;
uint64_t *aResultColsCpu;
long long *aPatFilteredCpu;

__global__ void getPatterns(ulong maxPatterns, uint64_t *aResultCols, int id,
		ulong bicsPerGpuPrevious, uint64_t *mInputData, long long *patFiltered,
		ulong totalPatterns, ulong patternsPerRun, int iter, ulong totalFor) {
	ulong idTh = blockIdx.x * blockDim.x + threadIdx.x;
	ulong pattern = idTh + (totalFor * (iter - 1)) + (id * bicsPerGpuPrevious)
			+ totalPatterns;
	if (idTh < patternsPerRun && pattern < maxPatterns) {
		long r1 = 0;
		long r2 = -1;
		long auxPat = pattern - rows + 1;
		if (auxPat < 0) {
			r2 = auxPat + rows;
		}
		for (ulong j = rows - 2; r2 == -1; j--) {
			auxPat = auxPat - j;
			r1++;
			if (auxPat < 0) {
				r2 = (j + auxPat) + (r1 + 1);
			}
		}

		if (r1 < rows && r2 < rows) {
			ulong totalOnes = 0;
			for (ulong j = 0; j < cols; j++) {
				uint64_t rAnd = *(mInputData + r1 * cols + j)
						& *(mInputData + r2 * cols + j);
				*(aResultCols + idTh * cols + j) = rAnd;
				while (rAnd) {
					if (rAnd & 1 == 1) {
						totalOnes++;
					}
					rAnd >>= 1;
				}
			}
			if (totalOnes >= mnc) {
				unsigned long long int current_val = atomicAdd(&numPatFiltered,
						1);
				*(patFiltered + current_val) = idTh;
			}
		}
	}
}

__global__ void getPatterns(ulong maxPatterns, uint32_t *aResultCols, int id,
		ulong bicsPerGpuPrevious, uint32_t *mInputData, long long *patFiltered,
		ulong totalPatterns, ulong patternsPerRun, int iter, ulong totalFor) {
	ulong idTh = blockIdx.x * blockDim.x + threadIdx.x;
	ulong pattern = idTh + (totalFor * (iter - 1)) + (id * bicsPerGpuPrevious)
			+ totalPatterns;
	if (idTh < patternsPerRun && pattern < maxPatterns) {
		long r1 = 0;
		long r2 = -1;
		long auxPat = pattern - rows + 1;
		if (auxPat < 0) {
			r2 = auxPat + rows;
		}
		for (ulong j = rows - 2; r2 == -1; j--) {
			auxPat = auxPat - j;
			r1++;
			if (auxPat < 0) {
				r2 = (j + auxPat) + (r1 + 1);
			}
		}
		if (r1 < rows && r2 < rows) {
			ulong totalOnes = 0;
			for (ulong j = 0; j < cols; j++) {
				uint32_t rAnd = *(mInputData + r1 * cols + j)
						& *(mInputData + r2 * cols + j);
				*(aResultCols + idTh * cols + j) = rAnd;
				while (rAnd) {
					if (rAnd & 1 == 1) {
						totalOnes++;
					}
					rAnd >>= 1;
				}
			}
			if (totalOnes >= mnc) {
				unsigned long long int current_val = atomicAdd(&numPatFiltered,
						1);
				*(patFiltered + current_val) = idTh;
			}
		}
	}
}

__global__ void generateBiclusters(uint64_t *aResultCols, int id,
		uint64_t *mInputData, long long *patFiltered, uint8_t *aResult,
		int iter, ulong totalFor) {
	ulong patternArray = (blockIdx.x * blockDim.x + threadIdx.x
			+ (totalFor * (iter - 1)));
	if (patternArray < numPatFiltered) {
		ulong pattern = *(patFiltered + patternArray);
		for (ulong row = 0; row < rows; row++) {
			bool bEqual = true;
			for (ulong k = 0; k < cols && bEqual; k++) {
				uint64_t rPattern = *(aResultCols + pattern * cols + k);
				if (((uint64_t) *(mInputData + row * cols + k) & rPattern)
						!= rPattern) {
					bEqual = false;
				}
			}
			if (bEqual) {
				*(aResult + pattern * rows + row) = 1;
			} else {
				*(aResult + pattern * rows + row) = 0;
			}
		}
	}
}

__global__ void generateBiclusters(uint32_t *aResultCols, int id,
		uint32_t *mInputData, long long *patFiltered, uint8_t *aResult,
		int iter, ulong totalFor) {
	ulong patternArray = (blockIdx.x * blockDim.x + threadIdx.x
			+ (totalFor * (iter - 1)));
	if (patternArray < numPatFiltered) {
		ulong pattern = *(patFiltered + patternArray);
		for (ulong row = 0; row < rows; row++) {
			bool bEqual = true;
			for (ulong k = 0; k < cols && bEqual; k++) {
				uint32_t rPattern = *(aResultCols + pattern * cols + k);
				if (((uint32_t) *(mInputData + row * cols + k) & rPattern)
						!= rPattern) {
					bEqual = false;
				}
			}
			if (bEqual) {
				*(aResult + pattern * rows + row) = 1;
			} else {
				*(aResult + pattern * rows + row) = 0;
			}
		}
	}
}

__global__ void generateBiclusters_no_out(uint64_t *aResultCols, int id,
		uint64_t *mInputData, long long *patFiltered, int iter,
		ulong totalFor) {
	ulong patternArray = (blockIdx.x * blockDim.x + threadIdx.x
			+ (totalFor * (iter - 1)));
	if (patternArray < numPatFiltered) {
		ulong pattern = *(patFiltered + patternArray);
		uint64_t numRows = 0;
		for (ulong row = 0; row < rows && numRows <= mnr; row++) {
			bool bEqual = true;
			for (ulong k = 0; k < cols && bEqual; k++) {
				uint64_t rPattern = *(aResultCols + pattern * cols + k);
				if (((uint64_t) *(mInputData + row * cols + k) & rPattern)
						!= rPattern) {
					bEqual = false;
				}
			}
			if (bEqual) {
				numRows++;
			}
		}
		if (numRows >= mnr) {
			atomicAdd(&totalBiclusters, 1);
		}
	}
}

__global__ void generateBiclusters_no_out(uint32_t *aResultCols, int id,
		uint32_t *mInputData, long long *patFiltered, int iter,
		ulong totalFor) {
	ulong patternArray = (blockIdx.x * blockDim.x + threadIdx.x
			+ (totalFor * (iter - 1)));
	if (patternArray < numPatFiltered) {
		ulong pattern = *(patFiltered + patternArray);
		uint32_t numRows = 0;
		for (ulong row = 0; row < rows && numRows < mnr; row++) {
			bool bEqual = true;
			for (ulong k = 0; k < cols && bEqual; k++) {
				uint32_t rPattern = *(aResultCols + pattern * cols + k);
				if (((uint32_t) *(mInputData + row * cols + k) & rPattern)
						!= rPattern) {
					bEqual = false;
				}
			}
			if (bEqual) {
				numRows++;
			}
		}
		if (numRows >= mnr) {
			atomicAdd(&totalBiclusters, 1);
		}
	}
}

// #######
// # CPU #
// #######

// CPU: GENERAL FUNCS
// ------------------

void introduceParameters(char **argv) {

	//PARAMETER AUTOMATIC: pattern Size.
	patternSize = sizeof(void *) * 8;

	// PARAMETER 1: is a one-column file with the names of all the rows, following the order they appear in the dataset. It is used to print information about the elements of final biclusters
	inputFile = (char *) malloc(sizeof(char) * 250);
	inputFile = argv[1];

	// PARAMETER 2: minimum number of rows allowed in a valid bicluster (MNR)
	cMnr = atoi(argv[2]);

	// PARAMETER 3: minimum number of columns allowed in a valid bicluster (MNC)
	cMnc = atoi(argv[3]);

	//PARAMETER 4: OUTPUT
	output = atoi(argv[4]);

	//PARAMETER 5: GPus number
	deviceCount = atoi(argv[5]);

}

void getNumPatterns() {
	maxPatterns = 0;
	for (int i = 0; i < cRows; i++) {
		for (int j = i + 1; j < cRows; j++) {
			maxPatterns++;
		}
	}
}

void prepareGpu1D(ulong lNumber) {
	int device;
	cudaGetDevice(&device);
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	lastBlocksGrid = 1;
	maxIteratorGPU = 0;
	maxThreadsPerBlock = lNumber; // Case 1: 0 < lNumber <= prop.maxThreadsPerBlock
	if (lNumber > prop.maxThreadsPerBlock) { // Case 2: lNumber > prop.maxThreadsPerBlock && Supported GPU in a for
		maxThreadsPerBlock = prop.maxThreadsPerBlock;
		maxBlocksPerGrid = lNumber / prop.maxThreadsPerBlock;
		lastBlocksGrid = lNumber / prop.maxThreadsPerBlock;
		if (lNumber % prop.maxThreadsPerBlock != 0) {
			maxBlocksPerGrid++;
			lastBlocksGrid++;
		}
		if (maxBlocksPerGrid > prop.maxGridSize[1]) { // Case 3: Not supported GPU with a for --> Split patterns in multiple for
			maxIteratorGPU = maxBlocksPerGrid / prop.maxGridSize[1];
			lastBlocksGrid = maxBlocksPerGrid
					- (maxIteratorGPU * prop.maxGridSize[1]);
			maxBlocksPerGrid = prop.maxGridSize[1];
		}
	}
}

// CPU: SPECIFIC FUNCS
// -------------------

//String (binary) to UINT64 (64 bits)
uint64_t binaryToDecimal64(const std::string& binary) {
	uint64_t decimal = 0;
	uint64_t p = 1;
	std::string::const_reverse_iterator iter;

	for (iter = binary.rbegin(); iter != binary.rend(); iter++) {
		if (*iter == '1')
			decimal += p;
		p *= 2;
	}

	return decimal;
}

//String (binary) to UINT32 (32 bits)
uint32_t binaryToDecimal32(const std::string& binary) {
	uint32_t decimal = 0;
	uint32_t p = 1;
	std::string::const_reverse_iterator iter;

	for (iter = binary.rbegin(); iter != binary.rend(); iter++) {
		if (*iter == '1')
			decimal += p;
		p *= 2;
	}

	return decimal;
}

//UINT64 to String (binary) (64 bits)
std::string decimalToBinary64(uint64_t decimal) {
	std::string binary;
	while (decimal > 0) {
		if (decimal % 2 == 0)
			binary += '0';
		else
			binary += '1';
		decimal /= 2;
	}
	std::reverse(binary.begin(), binary.end());
	return binary;
}

//UINT32 to String (binary) (32 bits)
std::string decimalToBinary32(uint32_t decimal) {
	std::string binary;
	while (decimal > 0) {
		if (decimal % 2 == 0)
			binary += '0';
		else
			binary += '1';
		decimal /= 2;
	}
	std::reverse(binary.begin(), binary.end());
	return binary;
}

//Read file (64 bits)
uint64_t* fileReader64() {

	// 1) Prepare to GPU: Allocate mArray
	cRows = 0;
	cCols = 0;
	uint64_t *mArray;

	// 2) Prepare ROWS from file
	vector<string> rowsArray_Aux;
	string line;
	ifstream myfile(inputFile);
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			rowsArray_Aux.push_back(line);
			cRows++;
		}
		myfile.close();

		// 2) Get columns number
		char *cstr_calc_column = strdup(rowsArray_Aux[0].c_str());
		for (int k = strlen(cstr_calc_column) - 1; k >= 0; k--) {
			if (cstr_calc_column[k] == '0' || cstr_calc_column[k] == '1') {
				cCols++;
			}
		}

		cColsTotal = cCols;
		if (cCols % patternSize == 0) {
			cCols = cCols / patternSize;
		} else {
			cCols = (cCols / patternSize) + 1;
		}

		mArray = (uint64_t *) malloc(cRows * cCols * sizeof(uint64_t)); //Store character ASCII

		// 3) Create mArray
		for (int j = 0; j < cRows; j++) {
			char *cstr_rows = strdup(rowsArray_Aux[j].c_str());
			string cWord = "";
			int contmArray = cCols - 1;
			int contcWord = 0;
			for (int k = strlen(cstr_rows) - 1; k >= 0; k--) {
				if (cstr_rows[k] == '0' || cstr_rows[k] == '1') {
					cWord = cstr_rows[k] + cWord;
					contcWord++;
					if (contcWord == patternSize) {
						uint64_t iNumber = binaryToDecimal64(cWord);
						*(mArray + j * cCols + contmArray) = iNumber;
						contcWord = 0;
						cWord = "";
						contmArray--;
					}
				}
			}

			//Last cell
			if (contcWord != 0) {
				uint64_t iNumber = binaryToDecimal64(cWord);
				*(mArray + j * cCols + contmArray) = iNumber;
			}
		}

		for (int i = 0; i < deviceCount; i++) {
			cudaSetDevice(i);
			cudaMemcpyToSymbol(*(&mnc), &cMnc, sizeof(int), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&mnr), &cMnr, sizeof(int), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&cols), &cCols, sizeof(ulong), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&rows), &cRows, sizeof(ulong), 0,
					cudaMemcpyHostToDevice);
		}
	} else {
		cout << "Unable to open file " << endl;
	}

	return mArray;
}

//Read file (32 bits)
uint32_t* fileReader32() {

	// 1) Prepare to GPU: Allocate mArray
	cRows = 0;
	cCols = 0;
	uint32_t *mArray;

	// 2) Prepare ROWS from file
	vector<string> rowsArray_Aux;
	string line;
	ifstream myfile(inputFile);
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			rowsArray_Aux.push_back(line);
			cRows++;
		}
		myfile.close();

		// 2) Get columns number
		char *cstr_calc_column = strdup(rowsArray_Aux[0].c_str());
		for (int k = strlen(cstr_calc_column) - 1; k >= 0; k--) {
			if (cstr_calc_column[k] == '0' || cstr_calc_column[k] == '1') {
				cCols++;
			}
		}

		cColsTotal = cCols;
		if (cCols % patternSize == 0) {
			cCols = cCols / patternSize;
		} else {
			cCols = (cCols / patternSize) + 1;
		}

		mArray = (uint32_t *) malloc(cRows * cCols * sizeof(uint32_t)); //Store character ASCII

		// 3) Create mArray
		for (int j = 0; j < cRows; j++) {
			char *cstr_rows = strdup(rowsArray_Aux[j].c_str());
			string cWord = "";
			int contmArray = cCols - 1;
			int contcWord = 0;
			for (int k = strlen(cstr_rows) - 1; k >= 0; k--) {
				if (cstr_rows[k] == '0' || cstr_rows[k] == '1') {
					cWord = cstr_rows[k] + cWord;
					contcWord++;
					if (contcWord == patternSize) {
						uint32_t iNumber = binaryToDecimal32(cWord);
						*(mArray + j * cCols + contmArray) = iNumber;
						contcWord = 0;
						cWord = "";
						contmArray--;
					}
				}
			}

			//Last cell
			if (contcWord != 0) {
				uint32_t iNumber = binaryToDecimal32(cWord);
				*(mArray + j * cCols + contmArray) = iNumber;
			}
		}

		for (int i = 0; i < deviceCount; i++) {
			cudaSetDevice(i);
			cudaMemcpyToSymbol(*(&mnc), &cMnc, sizeof(int), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&mnr), &cMnr, sizeof(int), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&cols), &cCols, sizeof(ulong), 0,
					cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(*(&rows), &cRows, sizeof(ulong), 0,
					cudaMemcpyHostToDevice);
		}
	} else {
		cout << "Unable to open file " << endl;
	}

	return mArray;
}

//Get Array Result (64 bits)
uint64_t* getArrayResult64(int id, ulong numPatterns, ulong sizeArray) {
	uint64_t *aResult;
	cudaSetDevice(id);
	cudaMalloc((void **) &aResult, numPatterns * sizeArray * sizeof(uint64_t));
	cudaMemset(aResult, 0, numPatterns * sizeArray * sizeof(uint64_t));
	return aResult;
}

//Get Array Result (32 bits)
uint32_t* getArrayResult32(int id, ulong numPatterns, ulong sizeArray) {
	uint32_t *aResult;
	cudaSetDevice(id);
	cudaMalloc((void **) &aResult, numPatterns * sizeArray * sizeof(uint32_t));
	cudaMemset(aResult, 0, numPatterns * sizeArray * sizeof(uint32_t));
	return aResult;
}

long long* getPatternsFiltered(int id, ulong numPatterns) {
	long long *patFiltered;
	cudaMalloc((void **) &patFiltered, numPatterns * sizeof(long long));
	cudaMemset(patFiltered, -1, numPatterns * sizeof(long long));
	return patFiltered;
}

unsigned long long int printResults(int id, ulong patternsPerRun,
		uint8_t *aResult, long long *aPatFilteredCpu, uint64_t *aResultColsCpu,
		unsigned long long int cpuNumPatFiltered,
		unsigned long long int totalBic) {
	ulong iTotalRows;
	ofstream myfile;
	string line;
	myfile.open("results_GPU" + to_string(id) + ".txt");
	myfile << "Rows;Cols" << "\n";

	uint8_t *aResultCpu = (uint8_t *) malloc(
			patternsPerRun * cRows * sizeof(uint8_t));
	cudaMemcpy(aResultCpu, aResult, patternsPerRun * cRows * sizeof(uint8_t),
			cudaMemcpyDeviceToHost);

	for (ulong r = 0; r < cpuNumPatFiltered; r++) {
		iTotalRows = 0;
		line = "";
		long long pattern = *(aPatFilteredCpu + r);
		for (ulong c = 0; c < cRows; c++) {
			if (*(aResultCpu + pattern * cRows + c) == 1) {
				line = line + to_string(c + 1) + ",";
				iTotalRows += 1;
			}
		}
		line.pop_back();
		line = line + ";";

		if (iTotalRows >= cMnr) {
			ulong firstOne = 0, contBit = 0, realPos;
			for (ulong c = 0; c < cCols; c++) {
				uint64_t colsReduce = *(aResultColsCpu + pattern * cCols + c);
				string s = decimalToBinary64(colsReduce);
				firstOne = (cColsTotal - s.length() - (patternSize * c)) + 1;
				for (char& ch : s) {
					if (ch == '1') {
						realPos = firstOne + contBit;
						line = line + to_string(realPos) + ",";
					}
					contBit++;
				}
			}
			line.pop_back();
			myfile << line << "\n";
			totalBic++;
		}
	}
	free(aResultCpu);
	myfile.close();
	return totalBic;
}

unsigned long long int printResults(int id, ulong patternsPerRun,
		uint8_t *aResult, long long *aPatFilteredCpu, uint32_t *aResultColsCpu,
		unsigned long long int cpuNumPatFiltered,
		unsigned long long int totalBic) {
	ulong iTotalRows;
	ofstream myfile;
	string line;
	myfile.open("results_GPU" + to_string(id) + ".txt");
	myfile << "Rows;Cols" << "\n";

	uint8_t *aResultCpu = (uint8_t *) malloc(
			patternsPerRun * cRows * sizeof(uint8_t));
	cudaMemcpy(aResultCpu, aResult, patternsPerRun * cRows * sizeof(uint8_t),
			cudaMemcpyDeviceToHost);

	for (ulong r = 0; r < cpuNumPatFiltered; r++) {
		iTotalRows = 0;
		line = "";
		long long pattern = *(aPatFilteredCpu + r);
		for (ulong c = 0; c < cRows; c++) {
			if (*(aResultCpu + pattern * cRows + c) == 1) {
				line = line + to_string(c + 1) + ",";
				iTotalRows += 1;
			}
		}
		line.pop_back();
		line = line + ";";

		if (iTotalRows >= cMnr) {
			ulong firstOne = 0, contBit = 0, realPos;
			for (ulong c = 0; c < cCols; c++) {
				uint64_t colsReduce = *(aResultColsCpu + pattern * cCols + c);
				string s = decimalToBinary32(colsReduce);
				firstOne = (cColsTotal - s.length() - (patternSize * c)) + 1;
				for (char& ch : s) {
					if (ch == '1') {
						realPos = firstOne + contBit;
						line = line + to_string(realPos) + ",";
					}
					contBit++;
				}
			}
			line.pop_back();
			myfile << line << "\n";
			totalBic++;
		}
	}
	free(aResultCpu);
	myfile.close();
	return totalBic;
}

void threadsPerDevice_64(int id, cudaStream_t s, ulong chunks,
		ulong bicsPerGpuPrevious, ulong patternsPerRun, uint64_t *mInputData,
		mutex *m) {
	cudaSetDevice(id);
	ulong totalPatterns = 0;
	unsigned long long int totalBic = 0, totalPatFiltered;

	for (ulong largeScale = 0; largeScale < chunks; largeScale++) {
		uint64_t *aResultCols = getArrayResult64(id, patternsPerRun, cCols);
		long long *patFiltered = getPatternsFiltered(id, patternsPerRun);

		// 1) Generate total patterns
		prepareGpu1D(patternsPerRun);
		for (int i = 1; i <= maxIteratorGPU; i++) {
			getPatterns<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(maxPatterns, aResultCols, id, bicsPerGpuPrevious, mInputData, patFiltered, totalPatterns, patternsPerRun, i,maxThreadsPerBlock*maxBlocksPerGrid);
		}
		getPatterns<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(maxPatterns, aResultCols, id, bicsPerGpuPrevious, mInputData, patFiltered, totalPatterns, patternsPerRun, maxIteratorGPU+1, maxThreadsPerBlock*maxBlocksPerGrid);

		unsigned long long int cpuNumPatFiltered;
		cudaMemcpyFromSymbol(&cpuNumPatFiltered, numPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);

		// 2) Remove duplicate patterns
		uint64_t *aResultColsCpu = (uint64_t *) malloc(
				patternsPerRun * cCols * sizeof(uint64_t));
		cudaMemcpy(aResultColsCpu, aResultCols,
				patternsPerRun * cCols * sizeof(uint64_t),
				cudaMemcpyDeviceToHost);
		long long *aPatFilteredCpu = (long long *) malloc(
				patternsPerRun * sizeof(long long));
		cudaMemcpy(aPatFilteredCpu, patFiltered,
				patternsPerRun * sizeof(long long), cudaMemcpyDeviceToHost);
		cudaFree(patFiltered);
		std::pair<std::set<vector<uint32_t>>::iterator, bool> ret;
		for (uint64_t i = 0; i < cpuNumPatFiltered; i++) {
			long long pat = *(aPatFilteredCpu + i);
			uint64_t *ptr = &aResultColsCpu[pat * cCols];
			vector<uint64_t> vec(ptr, ptr + cCols);
			vector<uint64_t> vec2 = vec;
			m->lock();
			if (setPatterns64.insert(vec2).second == false) {
				*(aPatFilteredCpu + i) = *(aPatFilteredCpu + cpuNumPatFiltered
						- 1);
				*(aPatFilteredCpu + cpuNumPatFiltered - 1) = -1;
				cpuNumPatFiltered--;
				i--;
			}
			m->unlock();
		}

		cudaMalloc((void **) &patFiltered, patternsPerRun * sizeof(long long));
		cudaMemcpy(patFiltered, aPatFilteredCpu,
				patternsPerRun * sizeof(long long), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(numPatFiltered, &cpuNumPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);

		// 3)  Generate biclusters
		uint8_t *aResult;
		if (output == 1) {
			cudaSetDevice(id);
			cudaMalloc((void **) &aResult,
					patternsPerRun * cRows * sizeof(uint8_t));
			cudaMemset(aResult, 0, patternsPerRun * cRows * sizeof(uint8_t));
		}

		prepareGpu1D(cpuNumPatFiltered);
		if (output == 1) {
			for (int i = 1; i <= maxIteratorGPU; i++) {
				generateBiclusters<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, aResult, i, maxBlocksPerGrid);
			}
			generateBiclusters<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, aResult, maxIteratorGPU+1, maxBlocksPerGrid);
		} else {
			for(int i=1; i <= maxIteratorGPU; i++) {
				generateBiclusters_no_out<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, i, maxBlocksPerGrid);
			}
			generateBiclusters_no_out<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, maxIteratorGPU+1, maxBlocksPerGrid);
		}

		// 4) Print results
		if (output == 1) {
			totalBic = printResults(id, patternsPerRun, aResult,
					aPatFilteredCpu, aResultColsCpu, cpuNumPatFiltered,
					totalBic);
		}

		if (output == 0) {
			cudaMemcpyFromSymbol(&totalBic, totalBiclusters,
					sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);
		}

		cudaMemcpyFromSymbol(&totalPatFiltered, numPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);
		totales += totalBic;
		totalBic = 0;
		totalPatFiltered = 0;
		cudaMemcpyToSymbol(totalBiclusters, &totalBic,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(numPatFiltered, &totalPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);
		free(aResultColsCpu);
		free(aPatFilteredCpu);
		cudaFree(aResult);
		cudaFree(aResultCols);
		cudaFree(patFiltered);
		totalPatterns += patternsPerRun;
	}
}

void threadsPerDevice_32(int id, cudaStream_t s, ulong chunks,
		ulong bicsPerGpuPrevious, ulong patternsPerRun, uint32_t *mInputData,
		mutex *m) {
	cudaSetDevice(id);
	ulong totalPatterns = 0;
	unsigned long long int totalBic = 0, totalPatFiltered;

	for (ulong largeScale = 0; largeScale < chunks; largeScale++) {
		uint32_t *aResultCols = getArrayResult32(id, patternsPerRun, cCols);
		long long *patFiltered = getPatternsFiltered(id, patternsPerRun);

		// 1) Generate total patterns
		prepareGpu1D(patternsPerRun);
		for (int i = 1; i <= maxIteratorGPU; i++) {
			getPatterns<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(maxPatterns, aResultCols, id, bicsPerGpuPrevious, mInputData, patFiltered, totalPatterns, patternsPerRun, i,maxThreadsPerBlock*maxBlocksPerGrid);
		}
		getPatterns<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(maxPatterns, aResultCols, id, bicsPerGpuPrevious, mInputData, patFiltered, totalPatterns, patternsPerRun, maxIteratorGPU+1, maxThreadsPerBlock*maxBlocksPerGrid);

		unsigned long long int cpuNumPatFiltered;
		cudaMemcpyFromSymbol(&cpuNumPatFiltered, numPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);

		// 2) Remove duplicate patterns
		uint32_t *aResultColsCpu = (uint32_t *) malloc(
				patternsPerRun * cCols * sizeof(uint32_t));
		cudaMemcpy(aResultColsCpu, aResultCols,
				patternsPerRun * cCols * sizeof(uint32_t),
				cudaMemcpyDeviceToHost);
		long long *aPatFilteredCpu = (long long *) malloc(
				patternsPerRun * sizeof(long long));
		cudaMemcpy(aPatFilteredCpu, patFiltered,
				patternsPerRun * sizeof(long long), cudaMemcpyDeviceToHost);
		cudaFree(patFiltered);
		std::pair<std::set<vector<uint32_t>>::iterator, bool> ret;
		for (uint32_t i = 0; i < cpuNumPatFiltered; i++) {
			long long pat = *(aPatFilteredCpu + i);
			uint32_t *ptr = &aResultColsCpu[pat * cCols];
			vector<uint32_t> vec(ptr, ptr + cCols);
			vector<uint32_t> vec2 = vec;
			m->lock();
			if (setPatterns32.insert(vec2).second == false) {
				*(aPatFilteredCpu + i) = *(aPatFilteredCpu + cpuNumPatFiltered
						- 1);
				*(aPatFilteredCpu + cpuNumPatFiltered - 1) = -1;
				cpuNumPatFiltered--;
				i--;
			}
			m->unlock();
		}
		cudaMalloc((void **) &patFiltered, patternsPerRun * sizeof(long long));
		cudaMemcpy(patFiltered, aPatFilteredCpu,
				patternsPerRun * sizeof(long long), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(numPatFiltered, &cpuNumPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);

		// 3)  Generate biclusters
		uint8_t *aResult;
		if (output == 1) {
			cudaSetDevice(id);
			cudaMalloc((void **) &aResult,
					patternsPerRun * cRows * sizeof(uint8_t));
			cudaMemset(aResult, 0, patternsPerRun * cRows * sizeof(uint8_t));
		}

		prepareGpu1D(cpuNumPatFiltered);
		if (output == 1) {
			for (int i = 1; i <= maxIteratorGPU; i++) {
				generateBiclusters<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, aResult, i, maxBlocksPerGrid);
			}
			generateBiclusters<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, aResult, maxIteratorGPU+1, maxBlocksPerGrid);
		} else {
			for(int i=1; i <= maxIteratorGPU; i++) {
				generateBiclusters_no_out<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, i, maxBlocksPerGrid);
			}
			generateBiclusters_no_out<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(aResultCols, id, mInputData, patFiltered, maxIteratorGPU+1, maxBlocksPerGrid);
		}

		// 4) Print biclusters
		if (output == 1) {
			totalBic = printResults(id, patternsPerRun, aResult,
					aPatFilteredCpu, aResultColsCpu, cpuNumPatFiltered,
					totalBic);
		}

		if (output == 0) {
			cudaMemcpyFromSymbol(&totalBic, totalBiclusters,
					sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);
		}

		cudaMemcpyFromSymbol(&totalPatFiltered, numPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);
		totales += totalBic;
		totalBic = 0;
		totalPatFiltered = 0;
		cudaMemcpyToSymbol(totalBiclusters, &totalBic,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(numPatFiltered, &totalPatFiltered,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);
		free(aResultColsCpu);
		free(aPatFilteredCpu);
		cudaFree(aResult);
		cudaFree(aResultCols);
		cudaFree(patFiltered);
		totalPatterns += patternsPerRun;
	}
}

void runAlgorithm_64() {

	// 1) Create inputData (Matrix)
	totales = 0;
	uint64_t *mArray = fileReader64();
	getNumPatterns();

	// 2) PREPARING LARGE-SCALE DATA: CHUNKS
	cudaStream_t s[deviceCount];
	thread threads[deviceCount];
	ulong chunks[deviceCount], patternsPerRun[deviceCount];
	ulong bicsPerGpu = maxPatterns / deviceCount;
	ulong restBiclustersLastGpu = maxPatterns % deviceCount;

	for (int i = 0; i < deviceCount; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&s[i]);
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		double availableMemory = ((3 * prop.totalGlobalMem) / 4
				- (cRows * cCols * sizeof(uint64_t))); //InputData
		double sizeResult = 0;
		if (output == 1) {
			sizeResult = (bicsPerGpu * cRows * sizeof(char));
		}
		double sizeResultCols = (bicsPerGpu * cCols * sizeof(uint64_t));
		double patFiltered = (bicsPerGpu * sizeof(long long));
		chunks[i] = ((sizeResult + sizeResultCols + patFiltered)
				/ availableMemory) + 1;
		patternsPerRun[i] = bicsPerGpu / chunks[i];
		if (bicsPerGpu % chunks[i] != 0) {
			patternsPerRun[i]++;
		}
		if (deviceCount > 1 && maxPatterns % deviceCount != 0
				&& i == deviceCount - 1) {
			patternsPerRun[i] += restBiclustersLastGpu;
		}
	}

	ulong bicsPerGpuPrevious = 0;
	mutex m;
	for (int i = 0; i < deviceCount; i++) {
		uint64_t *mInputData;
		cudaSetDevice(i);
		cudaMallocHost((void**) &mInputData, cRows * cCols * sizeof(uint64_t));
		cudaMemcpy(mInputData, mArray, cRows * cCols * sizeof(uint64_t),
				cudaMemcpyHostToDevice);
		if (i > 0) {
			bicsPerGpuPrevious += chunks[i - 1] * patternsPerRun[i - 1];
		}
		threads[i] = thread(threadsPerDevice_64, i, s[i], chunks[i],
				bicsPerGpuPrevious, patternsPerRun[i], mInputData, &m);
	}

	for (auto& th : threads) {
		th.join();
	}
}

void runAlgorithm_32() {

	// 1) Create inputData (Matrix)
	totales = 0;
	uint32_t *mArray = fileReader32();
	getNumPatterns();

	// 2) PREPARING LARGE-SCALE DATA: CHUNKS
	cudaStream_t s[deviceCount];
	thread threads[deviceCount];
	ulong chunks[deviceCount], patternsPerRun[deviceCount];
	ulong bicsPerGpu = maxPatterns / deviceCount;
	ulong restBiclustersLastGpu = maxPatterns % deviceCount;

	for (int i = 0; i < deviceCount; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&s[i]);
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		double availableMemory = ((3 * prop.totalGlobalMem) / 4
				- (cRows * cCols * sizeof(uint32_t))); //InputData
		double sizeResult = 0;
		if (output != 0) {
			sizeResult = (bicsPerGpu * cRows * sizeof(char));
		}
		double sizeResultCols = (bicsPerGpu * cCols * sizeof(uint32_t));
		double patFiltered = (bicsPerGpu * sizeof(long long));
		chunks[i] = ((sizeResult + sizeResultCols + patFiltered)
				/ availableMemory) + 1;
		patternsPerRun[i] = bicsPerGpu / chunks[i];
		if (bicsPerGpu % chunks[i] != 0) {
			patternsPerRun[i]++;
		}
		if (deviceCount > 1 && maxPatterns % deviceCount != 0
				&& i == deviceCount - 1) {
			patternsPerRun[i] += restBiclustersLastGpu;
		}
	}

	ulong bicsPerGpuPrevious = 0;
	mutex m;
	for (int i = 0; i < deviceCount; i++) {
		uint32_t *mInputData;
		cudaSetDevice(i);
		cudaMallocHost((void**) &mInputData, cRows * cCols * sizeof(uint32_t));
		cudaMemcpy(mInputData, mArray, cRows * cCols * sizeof(uint32_t),
				cudaMemcpyHostToDevice);
		if (i > 0) {
			bicsPerGpuPrevious += chunks[i - 1] * patternsPerRun[i - 1];
		}
		threads[i] = thread(threadsPerDevice_32, i, s[i], chunks[i],
				bicsPerGpuPrevious, patternsPerRun[i], mInputData, &m);
	}

	for (auto& th : threads) {
		th.join();
	}
}

/*
 ########
 # MAIN #
 ########
 */
int main(int argc, char** argv) {

	introduceParameters(argv);

	//We will work with a maximum size of 32 or 64 bits (patternSize)
	if (patternSize == 64) {
		runAlgorithm_64();
	} else {
		runAlgorithm_32();
	}

	// PRINT BITPAT
	printf("Resume:\n========================\n");
	printf("Dataset filename: %s\n", inputFile);
	printf("Dataset size (rows,columns): %d, %d\n", cRows, cCols);
	printf("GPUs devices: %d\n", deviceCount);
	printf("Pattern size: %lu\n", patternSize);
	printf("MNC value: %lu\n", cMnc);
	printf("MNR value: %lu\n", cMnr);

	printf("\nResults:\n========================\n");
	cout << "Biclusters: " << totales << endl;

	return cudaDeviceReset();
}
