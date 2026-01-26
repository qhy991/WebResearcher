"""
WebResearch - GPU Kernel优化专用检索Agent
"""

from .gpu_kernel_search_agent import (
    GPUKernelSearchAgent,
    SearchResult,
    SearchResponse
)

__version__ = "1.0.0"
__all__ = ["GPUKernelSearchAgent", "SearchResult", "SearchResponse"]
