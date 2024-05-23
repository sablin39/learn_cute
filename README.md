# Learn CuTe

This is a repo that records my journey to CuTe, which is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. CuTe provides Layout and Tensor objects that compactly package the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.

The core abstractions of CuTe are hierarchically multidimensional layouts which can be composed with data arrays to represent tensors. The representation of layouts is powerful enough to represent nearly everything we need to implement efficient dense linear algebra. Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

The official tutorials are in https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute .

## Repo Architecture


## How to run these things

First you should have cutlass installed according to [the guidance](https://github.com/NVIDIA/cutlass/tree/main?tab=readme-ov-file#building-cutlass). Remember to modify paths in `CMakeLists.txt` so the compiler know where you've installed them.
