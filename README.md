# HR政策问答系统

基于RAG（检索增强生成）的企业年假政策智能问答系统。

## 功能

- 离线 Embedding 模型，无需联网
- 支持多轮对话记忆
- 流式输出（SSE），逐字返回答案
- FastAPI 提供 REST API 接口

## 技术栈

- Python 3.11+
- LangChain 1.0（LCEL 链式调用）
- FAISS 向量检索
- Sentence-Transformers（多语言 Embedding）
- DeepSeek API（LLM）
- FastAPI（Web 服务）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt