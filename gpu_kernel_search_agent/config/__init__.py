"""
WebResearch configuration: domain-aware search profiles.
"""

from .search_domains import (
    CODE_QUALITY_PROFILE,
    GENERAL_PROFILE,
    GPU_KERNEL_DOMAINS,
    GPU_KERNEL_KEYWORDS,
    KERNEL_PROFILE,
    LPU_PROFILE,
    QUERY_CATEGORIES,
    QUALITY_INDICATORS,
    REVIEW_PROFILE,
    SEARCH_PROFILES,
    SearchDomainProfile,
    calculate_relevance,
    enhance_query,
    get_search_profile,
    normalize_domain_name,
)

__all__ = [
    "SearchDomainProfile",
    "SEARCH_PROFILES",
    "KERNEL_PROFILE",
    "GENERAL_PROFILE",
    "CODE_QUALITY_PROFILE",
    "REVIEW_PROFILE",
    "LPU_PROFILE",
    "GPU_KERNEL_DOMAINS",
    "GPU_KERNEL_KEYWORDS",
    "QUERY_CATEGORIES",
    "QUALITY_INDICATORS",
    "get_search_profile",
    "normalize_domain_name",
    "enhance_query",
    "calculate_relevance",
]
