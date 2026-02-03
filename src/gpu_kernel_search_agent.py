"""
GPU Kernel Optimization专用检索Agent
使用Tavily API进行智能检索
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

from tavily import AsyncTavilyClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# 尝试导入LLM库（可选）
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Ollama支持（使用aiohttp）
try:
    import aiohttp
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.search_domains import GPU_KERNEL_KEYWORDS, QUERY_CATEGORIES, QUALITY_INDICATORS

# 初始化rich console
console = Console()

# 配置日志 - 同时输出到控制台和文件
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# 创建文件处理器
log_file = log_dir / f"search_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 配置根日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 避免重复日志
logger.propagate = False


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    url: str
    content: str
    score: float = 0.0
    source: str = ""
    published_date: Optional[str] = None
    relevance: str = "medium"  # high, medium, low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "published_date": self.published_date,
            "relevance": self.relevance
        }


@dataclass
class SearchResponse:
    """搜索响应数据类"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    answer: str = ""
    total_results: int = 0
    search_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "answer": self.answer,
            "total_results": self.total_results,
            "search_time": self.search_time,
            "timestamp": self.timestamp
        }


class GPUKernelSearchAgent:
    """
    GPU Kernel优化专用检索Agent

    特点:
    1. 使用Tavily API进行高质量网络检索
    2. 针对GPU Kernel优化领域的专业检索
    3. 自动增强查询以获得更好的搜索结果
    4. 支持代理设置
    """

    def __init__(
        self,
        api_key: str,
        proxy: Optional[str] = None,
        max_results: int = 10,
        search_depth: str = "advanced",  # basic or advanced
        include_answer: bool = True,
        include_raw_content: bool = False,
        use_llm_answer: bool = False,  # 是否使用LLM生成详细答案
        llm_provider: str = "openai",  # "openai" or "anthropic"
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        """
        初始化搜索Agent

        Args:
            api_key: Tavily API密钥
            proxy: 代理地址 (如: http://10.20.5.43:7891)
            max_results: 最大返回结果数
            search_depth: 搜索深度 (basic/advanced)
            include_answer: 是否包含AI生成的答案
            include_raw_content: 是否包含原始内容
            use_llm_answer: 是否使用LLM生成详细答案（需要配置LLM API密钥）
            llm_provider: LLM提供商 ("openai" 或 "anthropic")
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.api_key = api_key
        self.proxy = proxy
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.use_llm_answer = use_llm_answer

        # 设置代理环境变量
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            logger.info(f"代理已设置: {proxy}")

        # 初始化Tavily客户端
        self.client = AsyncTavilyClient(api_key=api_key)
        
        # 初始化答案生成器（如果启用）
        self.answer_generator = None
        if use_llm_answer:
            self.answer_generator = AnswerGenerator(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                proxy=proxy
            )
            logger.info(f"已启用LLM详细答案生成器 ({llm_provider})")
        
        logger.info("GPU Kernel搜索Agent初始化完成")

    def _enhance_query(self, query: str) -> str:
        """
        增强查询以获得更好的GPU Kernel优化相关结果

        Args:
            query: 原始查询

        Returns:
            增强后的查询
        """
        # 自动添加CUDA/GPU上下文
        enhanced = query

        # 如果查询中已经没有这些关键词，添加上下文
        context_keywords = ['cuda', 'gpu', 'kernel', 'optimization', 'nvidia']
        query_lower = query.lower()

        if not any(kw in query_lower for kw in context_keywords):
            # 尝试推断查询类别
            for category, keywords in QUERY_CATEGORIES.items():
                if any(kw in query_lower for kw in keywords):
                    enhanced = f"{query} {category} GPU CUDA optimization"
                    break
            else:
                enhanced = f"{query} GPU CUDA kernel optimization"

        return enhanced

    def _calculate_relevance(self, result: Dict[str, Any]) -> str:
        """
        计算搜索结果的相关性

        Args:
            result: 搜索结果

        Returns:
            相关性等级 (high/medium/low)
        """
        url = result.get('url', '')
        content = (result.get('content', '') or '').lower()

        # 检查来源质量
        for domain in QUALITY_INDICATORS['high']:
            if domain in url:
                return 'high'

        # 检查内容相关性
        gpu_keywords = ['cuda', 'gpu', 'kernel', 'optimization', 'tensor core', 'shared memory']
        keyword_count = sum(1 for kw in gpu_keywords if kw in content)

        if keyword_count >= 3:
            return 'high'
        elif keyword_count >= 1:
            return 'medium'
        else:
            return 'low'

    def _parse_search_results(self, response: Dict[str, Any], query: str) -> SearchResponse:
        """
        解析Tavily API响应

        Args:
            response: Tavily API响应
            query: 原始查询

        Returns:
            解析后的搜索响应
        """
        search_response = SearchResponse(query=query)
        search_response.answer = response.get('answer', '')

        results = response.get('results', [])
        search_response.total_results = len(results)

        for result in results[:self.max_results]:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                source=result.get('source', 'unknown'),
                published_date=result.get('published_date'),
                relevance=self._calculate_relevance(result)
            )
            search_response.results.append(search_result)

        return search_response


class AnswerGenerator:
    """
    基于LLM的详细答案生成器
    整合多个搜索结果，生成更详细、结构化的答案
    """
    
    def __init__(
        self,
        provider: str = "openai",  # "openai", "anthropic", or "ollama"
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        proxy: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        初始化答案生成器
        
        Args:
            provider: LLM提供商 ("openai", "anthropic", 或 "ollama")
            api_key: API密钥（Ollama不需要）
            model: 模型名称（如 "gpt-4", "claude-3-opus", "llama2", "qwen2.5"）
            proxy: 代理地址
            ollama_base_url: Ollama API基础URL（默认: http://localhost:11434）
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.model = model
        self.proxy = proxy
        self.ollama_base_url = ollama_base_url
        
        # 初始化客户端
        self.client = None
        if provider == "ollama":
            # Ollama不需要API密钥，直接使用HTTP API
            if OLLAMA_AVAILABLE:
                self.client = "ollama"  # 标记为ollama
                self.model = model or "llama2"
                logger.info(f"使用Ollama，模型: {self.model}, URL: {self.ollama_base_url}")
            else:
                logger.warning("aiohttp未安装，Ollama功能不可用")
        elif not self.api_key:
            logger.warning(f"未提供{provider} API密钥，LLM答案生成功能将不可用")
        elif provider == "openai" and OPENAI_AVAILABLE:
            try:
                self.client = AsyncOpenAI(api_key=self.api_key)
                self.model = model or "gpt-4-turbo-preview"
            except Exception as e:
                logger.warning(f"初始化OpenAI客户端失败: {e}")
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            try:
                self.client = AsyncAnthropic(api_key=self.api_key)
                self.model = model or "claude-3-opus-20240229"
            except Exception as e:
                logger.warning(f"初始化Anthropic客户端失败: {e}")
        else:
            logger.warning(f"LLM provider {provider} 不可用，将使用Tavily默认答案")
    
    def _build_prompt(self, query: str, results: List[SearchResult]) -> str:
        """
        构建详细的提示词
        
        Args:
            query: 用户查询
            results: 搜索结果列表
            
        Returns:
            构建好的提示词
        """
        # 从环境变量读取可配置参数
        max_results = int(os.getenv('LLM_MAX_RESULTS', '5'))
        high_relevance_length = int(os.getenv('LLM_HIGH_RELEVANCE_LENGTH', '1000'))
        medium_relevance_length = int(os.getenv('LLM_MEDIUM_RELEVANCE_LENGTH', '500'))
        low_relevance_length = int(os.getenv('LLM_LOW_RELEVANCE_LENGTH', '300'))
        
        # 构建搜索结果上下文
        context_parts = []
        for idx, result in enumerate(results[:max_results], 1):
            # 根据相关性动态调整内容长度
            if result.relevance == 'high':
                content_length = high_relevance_length
            elif result.relevance == 'medium':
                content_length = medium_relevance_length
            else:
                content_length = low_relevance_length
            
            context_parts.append(
                f"【来源 {idx}】{result.title}\n"
                f"URL: {result.url}\n"
                f"内容摘要: {result.content[:content_length]}\n"
                f"相关性: {result.relevance}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""你是一个GPU Kernel优化领域的专家助手。请基于以下搜索结果，为用户的问题生成一个详细、结构化的答案。

用户问题: {query}

搜索结果:
{context}

请按照以下结构生成详细答案：

1. **概述** (2-3句话总结)
   - 简要回答用户的核心问题

2. **详细说明** (这是最重要的部分，需要详细展开)
   - 解释关键概念和原理
   - 提供具体的技术细节
   - 说明实现步骤或使用方法
   - 包含重要的注意事项

3. **实践建议**
   - 提供最佳实践
   - 常见问题和解决方案
   - 性能优化建议

4. **参考资源**
   - 列出最相关的资源链接（从搜索结果中选择）

要求：
- 答案要详细、准确、实用
- 使用专业但易懂的语言
- 如果涉及代码，提供具体示例
- 引用具体的搜索结果来源
- 如果搜索结果不够详细，可以基于专业知识进行补充说明
- 答案长度应该在500-1000字之间

请开始生成详细答案："""
        
        return prompt
    
    async def generate_detailed_answer(
        self,
        query: str,
        results: List[SearchResult]
    ) -> str:
        """
        基于搜索结果生成详细答案
        
        Args:
            query: 用户查询
            results: 搜索结果列表
            
        Returns:
            生成的详细答案
        """
        if not self.client:
            return ""
        
        try:
            prompt = self._build_prompt(query, results)
            system_prompt = "你是一个专业的GPU Kernel优化技术专家，擅长将复杂的技术概念解释得清晰易懂。"
            
            if self.provider == "ollama":
                # 使用Ollama HTTP API
                async with aiohttp.ClientSession() as session:
                    # Ollama API格式
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    payload = {
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 2000
                        }
                    }
                    
                    url = f"{self.ollama_base_url}/api/generate"
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            answer = result.get("response", "")
                        else:
                            error_text = await response.text()
                            logger.error(f"Ollama API错误 {response.status}: {error_text}")
                            return ""
            
            elif self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.content[0].text
            else:
                logger.warning(f"未知的provider: {self.provider}")
                return ""
            
            logger.info(f"使用 {self.provider} 生成了详细答案，长度: {len(answer)} 字符")
            return answer
            
        except Exception as e:
            logger.error(f"生成详细答案时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""


class GPUKernelSearchAgent:
    """
    GPU Kernel优化专用检索Agent

    特点:
    1. 使用Tavily API进行高质量网络检索
    2. 针对GPU Kernel优化领域的专业检索
    3. 自动增强查询以获得更好的搜索结果
    4. 支持代理设置
    5. 可选使用LLM生成详细答案
    """

    def __init__(
        self,
        api_key: str,
        proxy: Optional[str] = None,
        max_results: int = 10,
        search_depth: str = "advanced",  # basic or advanced
        include_answer: bool = True,
        include_raw_content: bool = False,
        use_llm_answer: bool = False,  # 是否使用LLM生成详细答案
        llm_provider: str = "openai",  # "openai", "anthropic", or "ollama"
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        初始化搜索Agent

        Args:
            api_key: Tavily API密钥
            proxy: 代理地址 (如: http://10.20.5.43:7891)
            max_results: 最大返回结果数
            search_depth: 搜索深度 (basic/advanced)
            include_answer: 是否包含AI生成的答案
            include_raw_content: 是否包含原始内容
            use_llm_answer: 是否使用LLM生成详细答案（需要配置LLM API密钥）
            llm_provider: LLM提供商 ("openai" 或 "anthropic")
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.api_key = api_key
        self.proxy = proxy
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.use_llm_answer = use_llm_answer

        # 设置代理环境变量
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            logger.info(f"代理已设置: {proxy}")

        # 初始化Tavily客户端
        self.client = AsyncTavilyClient(api_key=api_key)
        
        # 初始化答案生成器（如果启用）
        self.answer_generator = None
        if use_llm_answer:
            self.answer_generator = AnswerGenerator(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                proxy=proxy
            )
            logger.info(f"已启用LLM详细答案生成器 ({llm_provider})")
        
        logger.info("GPU Kernel搜索Agent初始化完成")

    def _enhance_query(self, query: str) -> str:
        """
        增强查询以获得更好的GPU Kernel优化相关结果

        Args:
            query: 原始查询

        Returns:
            增强后的查询
        """
        # 自动添加CUDA/GPU上下文
        enhanced = query

        # 如果查询中已经没有这些关键词，添加上下文
        context_keywords = ['cuda', 'gpu', 'kernel', 'optimization', 'nvidia']
        query_lower = query.lower()

        if not any(kw in query_lower for kw in context_keywords):
            # 尝试推断查询类别
            for category, keywords in QUERY_CATEGORIES.items():
                if any(kw in query_lower for kw in keywords):
                    enhanced = f"{query} {category} GPU CUDA optimization"
                    break
            else:
                enhanced = f"{query} GPU CUDA kernel optimization"

        return enhanced

    def _calculate_relevance(self, result: Dict[str, Any]) -> str:
        """
        计算搜索结果的相关性

        Args:
            result: 搜索结果

        Returns:
            相关性等级 (high/medium/low)
        """
        url = result.get('url', '')
        content = (result.get('content', '') or '').lower()

        # 检查来源质量
        for domain in QUALITY_INDICATORS['high']:
            if domain in url:
                return 'high'

        # 检查内容相关性
        gpu_keywords = ['cuda', 'gpu', 'kernel', 'optimization', 'tensor core', 'shared memory']
        keyword_count = sum(1 for kw in gpu_keywords if kw in content)

        if keyword_count >= 3:
            return 'high'
        elif keyword_count >= 1:
            return 'medium'
        else:
            return 'low'

    def _parse_search_results(self, response: Dict[str, Any], query: str) -> SearchResponse:
        """
        解析Tavily API响应

        Args:
            response: Tavily API响应
            query: 原始查询

        Returns:
            解析后的搜索响应
        """
        search_response = SearchResponse(query=query)
        search_response.answer = response.get('answer', '')

        results = response.get('results', [])
        search_response.total_results = len(results)

        for result in results[:self.max_results]:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                source=result.get('source', 'unknown'),
                published_date=result.get('published_date'),
                relevance=self._calculate_relevance(result)
            )
            search_response.results.append(search_result)

        return search_response

    async def search(self, query: str, enhance_query: bool = True) -> SearchResponse:
        """
        执行搜索

        Args:
            query: 搜索查询
            enhance_query: 是否增强查询

        Returns:
            搜索响应
        """
        import time
        start_time = time.time()

        # 增强查询
        search_query = self._enhance_query(query) if enhance_query else query
        console.print(f"[cyan]搜索查询:[/cyan] {search_query}")

        try:
            # 调用Tavily API
            response = await self.client.search(
                query=search_query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_images=False,
                include_image_descriptions=False
            )

            # 解析结果
            search_response = self._parse_search_results(response, query)
            search_response.search_time = time.time() - start_time

            # 如果启用了LLM答案生成器，生成更详细的答案
            if self.use_llm_answer and self.answer_generator and search_response.results:
                console.print("[cyan]正在生成详细答案...[/cyan]")
                try:
                    detailed_answer = await self.answer_generator.generate_detailed_answer(
                        query=query,
                        results=search_response.results
                    )
                    if detailed_answer:
                        # 将Tavily的简短答案和LLM的详细答案合并
                        if search_response.answer:
                            search_response.answer = f"{search_response.answer}\n\n---\n\n## 详细解答\n\n{detailed_answer}"
                        else:
                            search_response.answer = detailed_answer
                        logger.info("已生成详细答案")
                except Exception as e:
                    logger.warning(f"生成详细答案失败，使用Tavily默认答案: {e}")

            logger.info(f"搜索完成，找到 {search_response.total_results} 个结果，耗时 {search_response.search_time:.2f}s")

            return search_response

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    async def search_with_context(
        self,
        query: str,
        context: str,
        max_results: int = 5
    ) -> SearchResponse:
        """
        带上下文的搜索

        Args:
            query: 搜索查询
            context: 上下文信息
            max_results: 最大结果数

        Returns:
            搜索响应
        """
        enhanced_query = f"{query} {context}"
        console.print(f"[cyan]上下文搜索:[/cyan] {enhanced_query}")

        return await self.search(enhanced_query, enhance_query=False)

    def display_results(self, response: SearchResponse):
        """
        使用Rich库美化显示搜索结果

        Args:
            response: 搜索响应
        """
        console.print(f"\n[bold green]=== 搜索结果 ===[/bold green]")
        console.print(f"[cyan]查询:[/cyan] {response.query}")
        console.print(f"[cyan]耗时:[/cyan] {response.search_time:.2f}s")
        console.print(f"[cyan]结果数:[/cyan] {response.total_results}\n")

        # 显示AI答案
        if response.answer:
            console.print(Panel(response.answer, title="[bold]AI生成的答案[/bold]", border_style="blue"))

        # 创建结果表格
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("相关性", style="bold", width=8)
        table.add_column("标题", style="cyan", no_wrap=False)
        table.add_column("来源", style="yellow", width=15)
        table.add_column("URL", style="blue", width=40)

        for idx, result in enumerate(response.results, 1):
            relevance_color = {
                'high': 'green',
                'medium': 'yellow',
                'low': 'red'
            }.get(result.relevance, 'white')

            table.add_row(
                str(idx),
                f"[{relevance_color}]{result.relevance}[/{relevance_color}]",
                result.title[:60] + "..." if len(result.title) > 60 else result.title,
                result.source,
                result.url[:40] + "..." if len(result.url) > 40 else result.url
            )

        console.print(table)

        # 显示详细内容
        if response.results:
            console.print(f"\n[bold]详细内容:[/bold]")
            for idx, result in enumerate(response.results[:3], 1):  # 只显示前3个详情
                console.print(f"\n[cyan][{idx}] {result.title}[/cyan]")
                console.print(f"[dim]URL: {result.url}[/dim]")
                console.print(f"[white]{result.content[:300]}...[/white]\n")

    def save_results(self, response: SearchResponse, output_dir: str = "data"):
        """
        保存搜索结果到JSON文件

        Args:
            response: 搜索响应
            output_dir: 输出目录（相对于项目根目录）
        """
        # 确保输出目录是相对于项目根目录的
        if not os.path.isabs(output_dir):
            output_dir = project_root / output_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in response.query)
        filename = f"{timestamp}_{safe_query[:50]}.json"
        filepath = output_dir / filename

        # 保存结果
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)

        console.print(f"[green]结果已保存到:[/green] {filepath}")
        logger.info(f"搜索结果已保存到: {filepath}")

        return str(filepath)


async def main():
    """主函数 - 示例用法"""
    # API密钥
    API_KEY = "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"

    # 创建搜索Agent（不使用代理）
    agent = GPUKernelSearchAgent(
        api_key=API_KEY,
        proxy=None,  # 不使用代理
        max_results=10,
        search_depth="advanced",
        include_answer=True
    )

    # 示例查询
    query = "tensor core MMA instruction optimization for matrix multiplication"

    console.print(Panel.fit(
        "[bold cyan]GPU Kernel Optimization Search Agent[/bold cyan]\n"
        "Using Tavily API for intelligent web search",
        border_style="cyan"
    ))

    # 执行搜索
    response = await agent.search(query)

    # 显示结果
    agent.display_results(response)

    # 保存结果
    agent.save_results(response)


if __name__ == "__main__":
    asyncio.run(main())
