#!/bin/bash
# WebResearch 快速启动脚本

# 进入src目录
cd "$(dirname "$0")/src"

# 启动交互式CLI
python cli.py "$@"
