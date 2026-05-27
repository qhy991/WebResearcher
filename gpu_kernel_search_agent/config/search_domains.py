"""
Domain-aware search profiles for WebResearch.

Kernel GPU optimization remains the flagship profile (``kernel``), but callers
can select other profiles (``general``, ``code_quality``, ``lpu``, ``review``)
or pass a custom :class:`SearchDomainProfile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SearchDomainProfile:
    """Controls query enhancement and relevance scoring for one task domain."""

    name: str
    display_name: str = ""
    auto_enhance: bool = True
    context_keywords: Tuple[str, ...] = ()
    query_categories: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    category_enhance_template: str = "{query} {category}"
    default_enhance_suffix: str = ""
    relevance_keywords: Tuple[str, ...] = ()
    quality_indicators: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    preferred_domains: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "display_name",
            self.display_name or self.name.replace("_", " ").title(),
        )


# Legacy exports (kernel flagship domain) — kept for backward compatibility.
GPU_KERNEL_DOMAINS: Tuple[str, ...] = (
    "docs.nvidia.com",
    "developer.nvidia.com",
    "github.com/NVIDIA/cuda-samples",
    "github.com/NVIDIA",
    "github.com/NVIDIA/cublas",
    "github.com/NVIDIA/cutlass",
    "arxiv.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "pytorch.org",
    "tensorflow.org",
    "github.com",
    "stackoverflow.com",
    "developer.download.nvidia.com",
    "github.com/openai/triton",
    "tlcpack.ai",
    "tvm.apache.org",
)

GPU_KERNEL_KEYWORDS: Tuple[str, ...] = (
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
)

QUERY_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "performance": (
        "throughput",
        "latency",
        "TFLOPS",
        "bandwidth",
        "performance optimization",
        "kernel tuning",
    ),
    "memory": (
        "shared memory",
        "global memory",
        "memory coalescing",
        "memory alignment",
        "memory bank conflict",
    ),
    "architecture": (
        "tensor core",
        "warp scheduler",
        "SM",
        "HBM",
        "GPU architecture",
        "A100",
        "H100",
        "RTX 4090",
    ),
    "quantization": (
        "INT8",
        "FP16",
        "BF16",
        "W4A16",
        "W8A8",
        "quantization",
        "mixed precision",
    ),
    "libraries": (
        "cutlass",
        "cublas",
        "cudnn",
        "triton",
        "tilelang",
        "torch.compile",
    ),
}

QUALITY_INDICATORS: Dict[str, Tuple[str, ...]] = {
    "high": (
        "developer.nvidia.com",
        "docs.nvidia.com",
        "arxiv.org",
        "github.com/NVIDIA",
    ),
    "medium": (
        "github.com",
        "stackoverflow.com",
        "pytorch.org",
    ),
}

KERNEL_PROFILE = SearchDomainProfile(
    name="kernel",
    display_name="GPU Kernel",
    auto_enhance=True,
    context_keywords=("cuda", "gpu", "kernel", "optimization", "nvidia", "metal", "rocm", "hip"),
    query_categories=QUERY_CATEGORIES,
    category_enhance_template="{query} {category} GPU CUDA optimization",
    default_enhance_suffix="GPU CUDA kernel optimization",
    relevance_keywords=(
        "cuda",
        "gpu",
        "kernel",
        "optimization",
        "tensor core",
        "shared memory",
        "warp",
        "metal",
        "rocm",
    ),
    quality_indicators=QUALITY_INDICATORS,
    preferred_domains=GPU_KERNEL_DOMAINS,
)

GENERAL_PROFILE = SearchDomainProfile(
    name="general",
    display_name="General Engineering",
    auto_enhance=False,
    context_keywords=(),
    query_categories={},
    default_enhance_suffix="",
    relevance_keywords=("engineering", "software", "implementation", "design", "api"),
    quality_indicators={
        "high": ("arxiv.org", "github.com", "docs.python.org", "developer.mozilla.org"),
        "medium": ("stackoverflow.com", "medium.com", "dev.to"),
    },
)

CODE_QUALITY_PROFILE = SearchDomainProfile(
    name="code_quality",
    display_name="Code Quality",
    auto_enhance=True,
    context_keywords=("lint", "linter", "refactor", "test", "coverage", "static analysis", "type check"),
    query_categories={
        "testing": ("pytest", "unit test", "integration test", "mock", "fixture"),
        "style": ("pep8", "formatting", "ruff", "mypy", "typing"),
        "security": ("vulnerability", "cve", "sanitizer", "bounds check"),
    },
    category_enhance_template="{query} {category} software engineering",
    default_enhance_suffix="software engineering best practices",
    relevance_keywords=("test", "lint", "refactor", "coverage", "review", "quality", "ci"),
    quality_indicators={
        "high": ("docs.python.org", "pytest.org", "github.com"),
        "medium": ("stackoverflow.com", "realpython.com"),
    },
)

REVIEW_PROFILE = SearchDomainProfile(
    name="review",
    display_name="Code Review",
    auto_enhance=True,
    context_keywords=("review", "pull request", "pr", "diff", "merge request"),
    query_categories={
        "process": ("code review", "checklist", "best practice", "style guide"),
        "risk": ("security", "regression", "performance", "correctness"),
    },
    category_enhance_template="{query} {category} code review",
    default_enhance_suffix="code review best practices",
    relevance_keywords=("review", "pull request", "feedback", "maintainability", "readability"),
    quality_indicators={
        "high": ("google.github.io", "github.com"),
        "medium": ("stackoverflow.com", "martinfowler.com"),
    },
)

LPU_PROFILE = SearchDomainProfile(
    name="lpu",
    display_name="LPU / NPU",
    auto_enhance=True,
    context_keywords=("lpu", "npu", "accelerator", "onnx", "mlir", "tile", "systolic"),
    query_categories={
        "runtime": ("compiler", "scheduling", "memory hierarchy", "tiling"),
        "ops": ("matmul", "conv", "attention", "elementwise"),
    },
    category_enhance_template="{query} {category} NPU accelerator programming",
    default_enhance_suffix="LPU NPU accelerator operator programming",
    relevance_keywords=("lpu", "npu", "accelerator", "operator", "kernel", "tiling", "onnx"),
    quality_indicators={
        "high": ("arxiv.org", "github.com"),
        "medium": ("stackoverflow.com",),
    },
)

SEARCH_PROFILES: Dict[str, SearchDomainProfile] = {
    "kernel": KERNEL_PROFILE,
    "general": GENERAL_PROFILE,
    "code_quality": CODE_QUALITY_PROFILE,
    "review": REVIEW_PROFILE,
    "lpu": LPU_PROFILE,
    # Common aliases from KernelOwl domain packs / legacy names
    "smoke": GENERAL_PROFILE,
    "runtime_attempts": GENERAL_PROFILE,
    "capabilities": GENERAL_PROFILE,
    "robust_kbench": KERNEL_PROFILE,
    "kerneleval": KERNEL_PROFILE,
    "kernelevalplus": KERNEL_PROFILE,
}

_DEFAULT_PROFILE = GENERAL_PROFILE


def normalize_domain_name(domain: Optional[str]) -> str:
    text = str(domain or "").strip().lower().replace("-", "_")
    return text or "general"


def get_search_profile(domain: Optional[str] = None) -> SearchDomainProfile:
    key = normalize_domain_name(domain)
    return SEARCH_PROFILES.get(key, _DEFAULT_PROFILE)


def enhance_query(query: str, profile: Optional[SearchDomainProfile] = None) -> str:
    """Add domain context to a query when it lacks obvious domain signals."""
    prof = profile or _DEFAULT_PROFILE
    if not prof.auto_enhance:
        return query

    text = (query or "").strip()
    if not text:
        return query

    lowered = text.lower()
    if any(keyword in lowered for keyword in prof.context_keywords):
        return query

    for category, keywords in prof.query_categories.items():
        if any(keyword in lowered for keyword in keywords):
            return prof.category_enhance_template.format(query=text, category=category).strip()

    if prof.default_enhance_suffix:
        return f"{text} {prof.default_enhance_suffix}".strip()
    return query


def calculate_relevance(url: str, content: str, profile: Optional[SearchDomainProfile] = None) -> str:
    """Score a search hit as high / medium / low for the active profile."""
    prof = profile or _DEFAULT_PROFILE
    url_text = str(url or "")
    body = str(content or "").lower()

    for domain in prof.quality_indicators.get("high", ()):
        if domain in url_text:
            return "high"

    keyword_count = sum(1 for keyword in prof.relevance_keywords if keyword in body)
    if keyword_count >= 3:
        return "high"
    if keyword_count >= 1:
        return "medium"

    for domain in prof.quality_indicators.get("medium", ()):
        if domain in url_text:
            return "medium"

    return "low"
