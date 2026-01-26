#!/bin/bash
# WebResearch 快速启动脚本

# 设置代理
export https_proxy=http://10.20.5.43:7891
export http_proxy=http://10.20.5.43:7891

# 进入src目录
cd "$(dirname "$0")/src"

# 启动交互式CLI
python cli.py "$@"
