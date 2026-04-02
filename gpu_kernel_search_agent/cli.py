#!/usr/bin/env python3
"""
WebResearch - 交互式命令行界面
GPU Kernel优化专用检索Agent
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv未安装时跳过

from gpu_kernel_search_agent.gpu_kernel_search_agent import GPUKernelSearchAgent, console, Panel
from rich.prompt import Prompt
from rich import box


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   [bold cyan]WebResearch[/bold cyan] - [bold yellow]GPU Kernel Optimization[/bold yellow]     ║
║                                                               ║
║   [dim]Intelligent Search Agent Powered by Tavily API[/dim]      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(banner)


def print_examples():
    """打印示例查询"""
    examples = Panel(
        """
[bold yellow]示例查询:[/bold yellow]

[green]1. 性能优化类:[/green]
   [dim]• How to optimize shared memory usage in CUDA kernels[/dim]
   [dim]• Tensor Core MMA instruction best practices[/dim]
   [dim]• CUDA kernel occupancy calculation and optimization[/dim]

[green]2. 内存优化类:[/green]
   [dim]• Memory coalescing patterns for matrix multiplication[/dim]
   [dim]• Shared memory bank conflict resolution[/dim]

[green]3. 量化技术类:[/green]
   [dim]• INT8 quantization for GPU inference[/dim]
   [dim]• W4A16 quantization kernel implementation[/dim]

[green]4. 库和工具类:[/green]
   [dim]• CUTLASS GEMM optimization techniques[/dim]
   [dim]• Triton GPU kernel programming guide[/dim]
        """,
        title="[bold]查询示例[/bold]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(examples)


def print_commands():
    """打印可用命令"""
    commands = Panel(
        """
[bold yellow]可用命令:[/bold yellow]

[green]/q[/green]          - 退出程序
[green]/h[/green]          - 显示帮助
[green]/examples[/green]   - 显示查询示例
[green]/save[/green]       - 保存上次搜索结果
[green]/config[/green]     - 显示当前配置
[green]/ctx <context>[/green] - 使用上下文搜索

[bold cyan]直接输入问题开始搜索[/bold cyan]
        """,
        title="[bold]命令帮助[/bold]",
        border_style="yellow",
        box=box.ROUNDED
    )
    console.print(commands)


async def interactive_mode(
    api_key: str,
    proxy: Optional[str] = None,
    max_results: int = 10,
    search_depth: str = "advanced",
    auto_save: bool = False,
    output_dir: str = "data",
    use_llm_answer: bool = False,
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434"
):
    """
    交互式模式

    Args:
        api_key: Tavily API密钥
        proxy: 代理地址
        max_results: 最大结果数
        search_depth: 搜索深度
        auto_save: 是否自动保存
        output_dir: 输出目录
        use_llm_answer: 是否使用LLM生成详细答案
        llm_provider: LLM提供商
        llm_api_key: LLM API密钥
        llm_model: LLM模型名称
    """
    print_banner()

    # 创建Agent
    agent = GPUKernelSearchAgent(
        api_key=api_key,
        proxy=proxy,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
        use_llm_answer=use_llm_answer,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        ollama_base_url=ollama_base_url
    )

    # 显示配置
    config_panel = Panel(
        f"""
[bold cyan]API密钥:[/bold cyan] {api_key[:10]}...
[bold cyan]代理设置:[/bold cyan] {proxy or '未设置'}
[bold cyan]最大结果:[/bold cyan] {max_results}
[bold cyan]搜索深度:[/bold cyan] {search_depth}
[bold cyan]自动保存:[/bold cyan] {'是' if auto_save else '否'}
        """,
        title="[bold]当前配置[/bold]",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(config_panel)

    print_commands()

    last_response = None
    context = ""

    while True:
        try:
            # 获取用户输入
            prompt_str = "\n[bold cyan]请输入搜索查询[/bold cyan] (输入 /h 查看帮助): "
            if context:
                prompt_str = f"\n[bold yellow]当前上下文:[/bold yellow] {context}\n" + prompt_str

            user_input = Prompt.ask(prompt_str).strip()

            # 处理命令
            if user_input.lower() == '/q':
                console.print("[yellow]再见![/yellow]")
                break

            elif user_input.lower() == '/h':
                print_commands()

            elif user_input.lower() == '/examples':
                print_examples()

            elif user_input.lower() == '/config':
                console.print(config_panel)

            elif user_input.lower() == '/save':
                if last_response:
                    agent.save_results(last_response, output_dir)
                else:
                    console.print("[red]没有可保存的结果[/red]")

            elif user_input.lower().startswith('/ctx '):
                context = user_input[5:].strip()
                console.print(f"[green]上下文已设置为: {context}[/green]")

            elif user_input.startswith('/'):
                console.print(f"[red]未知命令: {user_input}[/red]")
                console.print("[dim]输入 /h 查看可用命令[/dim]")

            # 执行搜索
            elif user_input:
                console.print(f"\n[bold]正在搜索...[/bold]")

                if context:
                    response = await agent.search_with_context(user_input, context)
                else:
                    response = await agent.search(user_input)

                last_response = response

                # 显示结果
                agent.display_results(response)

                # 自动保存
                if auto_save:
                    agent.save_results(response, output_dir)

        except KeyboardInterrupt:
            console.print("\n[yellow]搜索已取消[/yellow]")
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


async def single_search_mode(
    query: str,
    api_key: str,
    proxy: Optional[str] = None,
    max_results: int = 10,
    search_depth: str = "advanced",
    save: bool = False,
    output_dir: str = "data",
    use_llm_answer: bool = False,
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434"
):
    """
    单次搜索模式

    Args:
        query: 搜索查询
        api_key: Tavily API密钥
        proxy: 代理地址
        max_results: 最大结果数
        search_depth: 搜索深度
        save: 是否保存结果
        output_dir: 输出目录
        use_llm_answer: 是否使用LLM生成详细答案
        llm_provider: LLM提供商
        llm_api_key: LLM API密钥
        llm_model: LLM模型名称
    """
    agent = GPUKernelSearchAgent(
        api_key=api_key,
        proxy=proxy,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
        use_llm_answer=use_llm_answer,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        ollama_base_url=ollama_base_url
    )

    console.print(f"[cyan]搜索查询:[/cyan] {query}\n")

    response = await agent.search(query)

    agent.display_results(response)

    # 默认保存搜索结果
    agent.save_results(response, output_dir)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="WebResearch - GPU Kernel优化专用检索Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python cli.py

  # 单次搜索
  python cli.py -q "tensor core optimization"

  # 使用代理
  python cli.py -q "CUDA kernel" --proxy http://10.20.5.43:7891

  # 保存结果
  python cli.py -q "shared memory" --save
        """
    )

    parser.add_argument(
        '-q', '--query',
        type=str,
        help='搜索查询 (不指定则进入交互模式)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default='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',
        help='Tavily API密钥'
    )

    parser.add_argument(
        '--proxy',
        type=str,
        default='http://10.20.5.43:7891',
        help='代理地址'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='最大返回结果数 (默认: 10)'
    )

    parser.add_argument(
        '--search-depth',
        type=str,
        choices=['basic', 'advanced'],
        default='advanced',
        help='搜索深度 (默认: advanced)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='保存搜索结果'
    )

    parser.add_argument(
        '--auto-save',
        action='store_true',
        help='交互模式下自动保存所有结果'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='结果保存目录 (默认: data)'
    )

    parser.add_argument(
        '--use-llm-answer',
        action='store_true',
        help='使用LLM生成详细答案（需要配置LLM API密钥）'
    )

    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama'],
        default=os.getenv('LLM_PROVIDER', 'openai'),
        help='LLM提供商 (默认: 从LLM_PROVIDER环境变量读取，或openai)'
    )

    parser.add_argument(
        '--ollama-base-url',
        type=str,
        default=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        help='Ollama API基础URL (默认: 从OLLAMA_BASE_URL环境变量读取，或http://localhost:11434)'
    )

    parser.add_argument(
        '--llm-api-key',
        type=str,
        default=None,
        help='LLM API密钥（优先使用命令行参数，否则从OPENAI_API_KEY或ANTHROPIC_API_KEY环境变量读取）'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='LLM模型名称（默认: 从LLM_MODEL环境变量读取，或根据provider使用默认值）'
    )

    args = parser.parse_args()
    
    # 从环境变量读取LLM API密钥（如果命令行未提供）
    if args.use_llm_answer and not args.llm_api_key:
        if args.llm_provider == 'openai':
            args.llm_api_key = os.getenv('OPENAI_API_KEY')
        elif args.llm_provider == 'anthropic':
            args.llm_api_key = os.getenv('ANTHROPIC_API_KEY')
        # Ollama不需要API密钥
    
    # 从环境变量读取LLM模型（如果命令行未提供）
    if args.use_llm_answer and not args.llm_model:
        if args.llm_provider == 'ollama':
            args.llm_model = os.getenv('OLLAMA_DEFAULT_MODEL', 'llama2')
        else:
            args.llm_model = os.getenv('LLM_MODEL')

    # 运行
    if args.query:
        # 单次搜索模式
        asyncio.run(single_search_mode(
            query=args.query,
            api_key=args.api_key,
            proxy=args.proxy,
            max_results=args.max_results,
            search_depth=args.search_depth,
            save=args.save,
            output_dir=args.output_dir,
            use_llm_answer=args.use_llm_answer,
            llm_provider=args.llm_provider,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            ollama_base_url=args.ollama_base_url
        ))
    else:
        # 交互式模式
        asyncio.run(interactive_mode(
            api_key=args.api_key,
            proxy=args.proxy,
            max_results=args.max_results,
            search_depth=args.search_depth,
            auto_save=args.auto_save,
            output_dir=args.output_dir,
            use_llm_answer=args.use_llm_answer,
            llm_provider=args.llm_provider,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            ollama_base_url=args.ollama_base_url
        ))


if __name__ == "__main__":
    main()
