# GSNet

## Prepare

- Download the assets from https://drive.google.com/file/d/1v4bmzibAKh4uXKIOVv2RCd-Iqy_0zFmi/view, unzip the assets to the root directory(/gsnet)

## Install

- Get the code (if not exists)

```bash
git clone https://github.com/rhett-chen/graspness_implementation.git
```

- Install **Minkowski Engine**

```bash
conda activate gsnet
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git # if not exist
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

- Install other packages

```bash
conda activate gsnet
# under gsnet folder
pip install -r requirements.txt
```

- Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet))

```bash
cd pointnet2
python setup.py install
```

- Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda))

```bash
cd knn
python setup.py install
```

- Install graspnetAPI for evaluation

```bash
git clone https://github.com/graspnet/graspnetAPI.git # if not exists
cd graspnetAPI
pip install .
```

# Troubleshooting

### pointnet2

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git # if not exists
cd Pointnet2_PyTorch
pip install -r requirements.txt 
# pip install -e .
pip install pointnet2_ops_lib/.
```

### knn, fatal error: THC/THC.h: No such file or directory

- comment `THC/THC.h`, `extern THCState *state`
- add `#include <ATen/cuda/ThrustAllocator.h>`
- modify `THCudaMalloc` to `c10::cuda::CUDACachingAllocator::raw_alloc：`
  //float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));
  float *dist_dev = (float*) c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));
- modify THCudaFree to c10::cuda::CUDACachingAllocator::raw_delete
  //THCudaFree(state, dist_dev);
  c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
- note THError("aborting");

### graspnetAPI

ERROR: Could not build wheels for cvxopt, which is required to install pyproject.toml-based projects

```bash
conda install -c conda-forge cvxopt
```
