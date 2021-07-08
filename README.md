# gBiBit: A multi-GPU biclustering algorithm for binary datasets
https://www.sciencedirect.com/science/article/pii/S0743731520303701

## Abstract
Graphichs Processing Units technology (GPU) and CUDA architecture are one of the most used options to adapt machine learning techniques to the huge amounts of complex data that are currently generated. Biclustering techniques are useful for discovering local patterns in datasets. Those of them that have been implemented to use GPU resources in parallel have improved their computational performance. However, this fact does not guarantee that they can successfully process large datasets. There are some important issues that must be taken into account, like the data transfers between CPU and GPU memory or the balanced distribution of workload between the GPU resources.
In this work, a GPU version of one of the fastest biclustering solutions, BiBit, is presented. This implementation, named gBiBit, has been designed to take full advantage of the computational resources offered by GPU devices. Either using a single GPU device or in its multi-GPU mode, gBiBit is the only binary biclustering algorithm that is able to process large datasets. The experimental results have shown that gBiBit improves the computational performance of BiBit and an early GPU version, called CUBiBit.

## Compilation
Go to gBiBit folder and executes the following commands to compile:
```
nvcc -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "src/BiBit.d" "src/BiBit.cu"
nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "src/BiBit.o" "src/BiBit.cu"
nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -link -o "gBiBit" ./src/BiBit.o
```


## Execution
### 1. Input parameters
1. inputFile (Character string): Absolute path of the input dataset file.
2. cMnr (integer number): Minimum number of rows allowed to build a final bicluster.
3. cMnc (integer number): Minimum number of columns allowed to build a final bicluster.
4. output (Value 1 or 0):
  - Value 1 = Shows the number of biclusters and final biclusters.
  - Value 0 = It shows only the number of biclusters generated.
5. deviceCount (integer number): Number of GPU devices you want to use.

### 2. Execute
_./BiBit [inputFile] [cMnr] [cMnc] [output] [deviceCount]_

```
./gBiBit /home/MyUser/Samples/Dataset.csv 3 3 0 2
```

## Authors
* [Aurelio Lopez-Fernandez](mailto:alopfer1@upo.es) - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Domingo Rodriguez-Baena - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Francisco Gomez Vela - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Federico Divina - [Data Science & Big Data Research Lab (Pablo de Olavide University)](https://datalab.upo.es/)
* Miguel Garcia-Torres - [Data Science & Big Data Research Lab (Pablo de Olavide University)](https://datalab.upo.es/)

## Contact
If you have comments or questions, or if you would like to contribute to the further development of gBiBit, please send us an email at alopfer1@upo.es

## License
This projected is licensed under the terms of the GNU General Public License v3.0.
