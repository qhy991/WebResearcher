# WebResearch - GPU Kernel优化专用检索Agent

基于Tavily API的智能网络检索工具，专注于GPU Kernel优化领域。

## 功能特点

- **专业领域搜索**: 针对GPU Kernel优化、CUDA编程、Tensor Core等方向进行优化
- **智能查询增强**: 自动为查询添加相关上下文，提高搜索质量
- **相关性评分**: 自动评估搜索结果的相关性和来源质量
- **AI生成答案**: 利用Tavily的AI能力生成综合性答案
- **精美终端展示**: 使用Rich库提供美观的命令行界面
- **结果持久化**: 支持将搜索结果保存为JSON格式

## 项目结构

```
WebResearch/
├── src/
│   ├── gpu_kernel_search_agent.py  # 核心搜索Agent
│   └── cli.py                       # 命令行界面
├── config/
│   ├── __init__.py
│   └── search_domains.py            # 搜索域配置
├── data/                            # 搜索结果保存目录
├── logs/                            # 日志目录
├── requirements.txt
├── .env.template
└── README.md
```

## 安装

### 1. 创建虚拟环境

```bash
cd WebResearch
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.template` 为 `.env`:

```bash
cp .env.template .env
```

编辑 `.env` 文件，设置你的API密钥和代理:

```bash
TAVILY_API_KEY=your_api_key_here
HTTP_PROXY=http://10.20.5.43:7891
HTTPS_PROXY=http://10.20.5.43:7891
```

## 使用方法

### 交互式模式

```bash
cd src
python cli.py
```

在交互式模式下，你可以:
- 直接输入问题进行搜索
- 使用 `/h` 查看帮助
- 使用 `/examples` 查看查询示例
- 使用 `/save` 保存上次搜索结果
- 使用 `/q` 退出程序

### 单次搜索模式

```bash
cd src
python cli.py -q "tensor core optimization techniques"
```

### 使用代理

```bash
python cli.py -q "CUDA kernel" --proxy http://10.20.5.43:7891
```

### 保存搜索结果

```bash
python cli.py -q "shared memory optimization" --save
```

### 完整参数示例

```bash
python cli.py \
  -q "matrix multiplication optimization" \
  --max-results 15 \
  --search-depth advanced \
  --save \
  --output-dir ../data
```

### 实际运行示例

以下是一个完整的运行示例，展示如何使用CUTLASS编写w8a8 GEMM CUDA kernel的搜索结果:

```bash
python cli.py -q "how to use CUTLASS to write a w8a8 GEMM CUDA kernel" --proxy http://10.20.5.43:7891
```

**运行输出:**

```
2026-01-26 09:51:21,171 - gpu_kernel_search_agent - INFO - 代理已设置: http://10.20.5.43:7891
2026-01-26 09:51:21,171 - gpu_kernel_search_agent - INFO - GPU Kernel搜索Agent初始化完成
搜索查询: how to use CUTLASS to write a w8a8 GEMM CUDA kernel

搜索查询: how to use CUTLASS to write a w8a8 GEMM CUDA kernel
2026-01-26 09:51:26,108 - gpu_kernel_search_agent - INFO - 搜索完成，找到 10 个结果，耗时 4.93s

=== 搜索结果 ===
查询: how to use CUTLASS to write a w8a8 GEMM CUDA kernel
耗时: 4.93s
结果数: 10

╭───────────────────────────────────────────────────────────────────────────────────────────────── AI生成的答案 ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ To write a w8a8 GEMM CUDA kernel with CUTLASS, include the CUTLASS header, define the GEMM operation, and launch it on the GPU. Use C++17 or greater for compilation.                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
┏━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ 相关性   ┃ 标题                                                            ┃ 来源            ┃ URL                                      ┃
┡━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ high     │ Quickstart — NVIDIA CUTLASS Documentation                       │ unknown         │ https://docs.nvidia.com/cutlass/latest/… │
│ 2   │ high     │ Efficient GEMM in CUDA — NVIDIA CUTLASS Documentation           │ unknown         │ https://docs.nvidia.com/cutlass/latest/… │
│ 3   │ high     │ NVIDIA/cutlass: CUDA Templates and Python DSLs for ... - Git... │ unknown         │ https://github.com/NVIDIA/cutlass        │
│ 4   │ high     │ CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelin... │ unknown         │ https://research.colfax-intl.com/cutlas… │
│ 5   │ medium   │ Build and Develop CUTLASS CUDA Kernels - Lei Mao's Log Book     │ unknown         │ https://leimao.github.io/blog/Build-Dev… │
│ 6   │ high     │ [PDF] Developing CUDA Kernels for Accelerated Matrix Multipl... │ unknown         │ https://research.colfax-intl.com/wp-con… │
│ 7   │ high     │ CUTLASS: A CUDA C++ Template Library for Accelerating Deep .... │ unknown         │ https://www.youtube.com/watch?v=PWWOGrL… │
│ 8   │ medium   │ How can I use Cutlass for my custom MMA operation? - Reddit     │ unknown         │ https://www.reddit.com/r/CUDA/comments/… │
│ 9   │ high     │ Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM... │ unknown         │ https://arxiv.org/html/2509.01229v1      │
│ 10  │ high     │ Advanced Matrix Multiplication Optimization on NVIDIA GPUs      │ unknown         │ https://salykova.github.io/sgemm-gpu     │
└─────┴──────────┴─────────────────────────────────────────────────────────────────┴─────────────────┴──────────────────────────────────────────┘

详细内容:

[1] Quickstart — NVIDIA CUTLASS Documentation
URL: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
## Using CUTLASS within other applications#

Applications should list `/include` within their include paths. They must be compiled as C++17 or greater.

Example: print the contents of a variable storing half-precision data.

```
 #include   #include   #include   #include   int  main()  { cutlass:: h...


[2] Efficient GEMM in CUDA — NVIDIA CUTLASS Documentation
URL: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
CUTLASS implements parallel reductions across threadblocks by partitioning the GEMM K dimension and launching an additional set of threadblocks for each partition. Consequently, we refer to this strategy within
CUTLASS as "parallel reduction splitK." The "parallel reduction splitK" strategy requires...


[3] NVIDIA/cutlass: CUDA Templates and Python DSLs for ... - GitHub
URL: https://github.com/NVIDIA/cutlass
To this rich ecosystem of C++ based kernel programming abstractions, CUTLASS 4 adds CUTLASS DSLs. These are Python native interfaces for writing high-performance CUDA kernels based on core CUTLASS and CuTe 
concepts without any performance compromises. This allows for a much smoother learning curve, ...
```

## 代码示例

### Python API使用

```python
import asyncio
from gpu_kernel_search_agent import GPUKernelSearchAgent

async def search_example():
    # 创建Agent
    agent = GPUKernelSearchAgent(
        api_key="your_api_key",
        proxy="http://10.20.5.43:7891",
        max_results=10,
        search_depth="advanced"
    )

    # 执行搜索
    response = await agent.search("tensor core MMA instruction")

    # 显示结果
    agent.display_results(response)

    # 保存结果
    agent.save_results(response)

# 运行
asyncio.run(search_example())
```

### 带上下文的搜索

```python
# 在特定上下文中搜索
response = await agent.search_with_context(
    query="optimization techniques",
    context="CUDA matrix multiplication H100 GPU"
)
```

## 搜索领域

该Agent针对以下领域进行了优化:

- **CUDA编程**: Kernel开发、内存管理、并发执行
- **GPU架构**: Tensor Core、SM、Warp、流处理器
- **性能优化**: Throughput、Latency、Occupancy、内存合并
- **量化技术**: INT8、FP16、BF16、W4A16、W8A8
- **优化库**: CUTLASS、cuBLAS、Triton、TileLang

## 查询示例

```
# 性能优化
How to optimize shared memory usage in CUDA kernels
Tensor Core MMA instruction best practices
CUDA kernel occupancy calculation and optimization

# 内存优化
Memory coalescing patterns for matrix multiplication
Shared memory bank conflict resolution

# 量化技术
INT8 quantization for GPU inference
W4A16 quantization kernel implementation

# 库和工具
CUTLASS GEMM optimization techniques
Triton GPU kernel programming guide
```

## 配置说明

### 环境变量配置

所有配置都可以通过 `.env` 文件或环境变量设置。复制 `.env.template` 为 `.env` 并编辑：

```bash
cp .env.template .env
```

### 基础配置

| 参数 | 环境变量 | 说明 | 默认值 |
|------|---------|------|--------|
| `api_key` | `TAVILY_API_KEY` | Tavily API密钥 | 必填 |
| `proxy` | `HTTP_PROXY` / `HTTPS_PROXY` | HTTP代理地址 | 可选 |
| `max_results` | `MAX_RESULTS` | 最大返回结果数 | 10 |
| `search_depth` | `SEARCH_DEPTH` | 搜索深度 (basic/advanced) | advanced |
| `include_answer` | - | 是否包含AI生成的答案 | True |
| `include_raw_content` | `INCLUDE_RAW_CONTENT` | 是否包含原始内容 | False |

### LLM答案生成配置

| 参数 | 环境变量 | 说明 | 默认值 |
|------|---------|------|--------|
| `use_llm_answer` | `USE_LLM_ANSWER` | 是否使用LLM生成详细答案 | false |
| `llm_provider` | `LLM_PROVIDER` | LLM提供商 (openai/anthropic/ollama) | openai |
| `llm_api_key` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | LLM API密钥 | - |
| `llm_model` | `LLM_MODEL` | LLM模型名称 | 根据provider默认 |
| `ollama_base_url` | `OLLAMA_BASE_URL` | Ollama API基础URL | http://localhost:11434 |
| `ollama_default_model` | `OLLAMA_DEFAULT_MODEL` | Ollama默认模型 | Qwen2.5-Coder:14B |

### LLM答案生成可调整参数

这些参数控制发送给LLM的内容量和质量：

| 参数 | 环境变量 | 说明 | 默认值 |
|------|---------|------|--------|
| `llm_max_results` | `LLM_MAX_RESULTS` | 发送给LLM的结果数量 | 5 |
| `llm_content_length` | `LLM_CONTENT_LENGTH` | 每个结果发送给LLM的字符数（已废弃，使用下面的分级配置） | 500 |
| `llm_high_relevance_length` | `LLM_HIGH_RELEVANCE_LENGTH` | 高相关性结果的内容长度 | 1000 |
| `llm_medium_relevance_length` | `LLM_MEDIUM_RELEVANCE_LENGTH` | 中等相关性结果的内容长度 | 500 |
| `llm_low_relevance_length` | `LLM_LOW_RELEVANCE_LENGTH` | 低相关性结果的内容长度 | 300 |

**优化建议：**
- 增加 `LLM_MAX_RESULTS` 可以包含更多搜索结果，但会增加token消耗
- 增加 `LLM_HIGH_RELEVANCE_LENGTH` 可以让高相关性结果提供更多细节
- 根据你的LLM模型上下文窗口大小调整这些参数

## 输出格式

搜索结果包含以下字段:

- `title`: 结果标题
- `url`: 来源链接
- `content`: 内容摘要
- `score`: 相关性得分
- `source`: 来源网站
- `relevance`: 相关性等级 (high/medium/low)
- `answer`: AI生成的综合答案

## 许可证

MIT License
