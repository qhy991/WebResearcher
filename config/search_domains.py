"""
GPU Kernel Optimization专用搜索域配置
"""

# GPU Kernel优化相关的专业域名和搜索源
GPU_KERNEL_DOMAINS = [
    # NVIDIA官方文档
    "docs.nvidia.com",
    "developer.nvidia.com",

    # CUDA相关
    "github.com/NVIDIA/cuda-samples",
    "github.com/NVIDIA",

    # 高性能计算库
    "github.com/NVIDIA/cublas",
    "github.com/NVIDIA/cutlass",
    "github.com/NVIDIA/M cutlass",

    # 论文和技术博客
    "arxiv.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",

    # 深度学习框架
    "pytorch.org",
    "tensorflow.org",

    # GPU优化社区
    "github.com",
    "stackoverflow.com",
    "developer.download.nvidia.com",

    # 张量编译
    "github.com/openai/triton",
    "tlcpack.ai",
    "tvm.apache.org",
]

# GPU Kernel优化关键词
GPU_KERNEL_KEYWORDS = [
    "CUDA kernel optimization",
    "GPU matrix multiplication",
    "shared memory optimization",
    "warp optimization",
    "tensor core",
    "MMA instruction",
    "CUDA cooperative groups",
    "streaming multiprocessor",
    "occupancy optimization",
    "memory coalescing",
    "PTX instruction",
    "SASS assembly",
    "warp-level primitive",
    "cutlass gemm",
    "cublas gemm",
    "triton GPU",
    "tilelang",
]

# 常见问题类别
QUERY_CATEGORIES = {
    "performance": [
        "throughput", "latency", "TFLOPS", "bandwidth",
        "performance optimization", "kernel tuning"
    ],
    "memory": [
        "shared memory", "global memory", "memory coalescing",
        "memory alignment", "memory bank conflict"
    ],
    "architecture": [
        "tensor core", "warp scheduler", "SM", "HBM",
        "GPU architecture", "A100", "H100", "RTX 4090"
    ],
    "quantization": [
        "INT8", "FP16", "BF16", "W4A16", "W8A8",
        "quantization", "mixed precision"
    ],
    "libraries": [
        "cutlass", "cublas", "cudnn", "triton",
        "tilelang", "torch.compile"
    ]
}

# 搜索结果质量评分关键词
QUALITY_INDICATORS = {
    "high": [
        "developer.nvidia.com",
        "docs.nvidia.com",
        "arxiv.org",
        "github.com/NVIDIA"
    ],
    "medium": [
        "github.com",
        "stackoverflow.com",
        "pytorch.org"
    ]
}
