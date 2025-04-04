#include "GPUMGARDCompressor.hpp"
#include "ProgressiveCompressor.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

// Modify when using different precision
using PRECISION = double;

PRECISION *readFile(const std::string &filename, size_t &numElements)
{
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        numElements = 0;
        return nullptr;
    }
    std::streamsize fileSize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (fileSize % sizeof(PRECISION) != 0)
    {
        std::cerr << "File size is not a multiple of PRECISION size." << std::endl;
        numElements = 0;
        return nullptr;
    }

    numElements = fileSize / sizeof(PRECISION);
    PRECISION *data = new PRECISION[numElements];
    if (!infile.read(reinterpret_cast<char *>(data), fileSize))
    {
        std::cerr << "Error reading file: " << filename << std::endl;
        delete[] data;
        numElements = 0;
        return nullptr;
    }
    return data;
}

int main()
{
    std::string filename =
        "/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/density.d64";
    size_t numElements = 384 * 384 * 256;
    PRECISION *data = readFile(filename, numElements);
    if (data == nullptr)
    {
        std::cerr << "Failed to read file." << std::endl;
        return 1;
    }

    mgard_x::DIM D = 3;
    std::vector<mgard_x::SIZE> shape = {384, 384, 256};
    const int nComponents = 10;
    double toleranceList[nComponents] = {5, 2.5, 1.25, 0.625, 0.3225,
                                         0.2, 0.1, 0.05, 0.01, 0.001};

    GPUMGARDCompressor<PRECISION> mgardCompressor;
    ProgressiveCompressor<PRECISION> progComp(&mgardCompressor);

    bool ok = progComp.compressData(D, shape, data, toleranceList, nComponents);
    if (!ok)
    {
        cerr << "Progressive compression failed." << endl;
        delete[] data;
        return 1;
    }

    double desiredErrorBound = 0.1;
    int reqComps = progComp.requestComponentsForBound(desiredErrorBound);
    cout << "For error bound " << desiredErrorBound << ", load " << reqComps << " components." << endl;

    // Reconstruct data based on reqComps
    PRECISION *reconstructed = progComp.reconstructData(reqComps);
    if (!reconstructed)
    {
        cerr << "Progressive reconstruction failed." << endl;
        delete[] data;
        return 1;
    }
    cout << "Reconstruction succeeded with " << reqComps << " components." << endl;

    delete[] data;
    delete[] reconstructed;
    return 0;
}
