"""
WebResearch - domain-aware web search (GPU kernel is the flagship profile).
"""

from .config.search_domains import (
    SearchDomainProfile,
    get_search_profile,
    enhance_query,
    calculate_relevance,
)
from .gpu_kernel_search_agent import (
    GPUKernelSearchAgent,
    SearchResult,
    SearchResponse,
)

# Backward-compatible alias
DomainAwareSearchAgent = GPUKernelSearchAgent

__version__ = "1.1.0"
__all__ = [
    "GPUKernelSearchAgent",
    "DomainAwareSearchAgent",
    "SearchResult",
    "SearchResponse",
    "SearchDomainProfile",
    "get_search_profile",
    "enhance_query",
    "calculate_relevance",
]
