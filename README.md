# Multi-Component-Compression-Framework

| Compressor Type | Version |
| :-------------: | ------: |
|      MGARD      |   1.5.2 |
|       ZFP       |   1.0.0 |
|       SZ        |  v3.1.7 |

To build the framework:
```
chmod +x build_scripts/build_cuda_hooper.sh
./build_script.sh 4
```
The building process may take about 30 mins.  
After build, zfp and sz will be installed at  
```/install-cuda-hooper ```  
and mgard will be installed at  
```build-cuda-hooper/mgard/src/install-cuda-ampere```  

To run test code:  
```
mkdir build && cd build
cmake ..
cmake --build .
./ProgressiveCompression
```