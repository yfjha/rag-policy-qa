#!/bin/bash
# 启动后端 (FastAPI)
uvicorn fastapi_rag:app --host 0.0.0.0 --port 8000 &
# 启动前端 (Streamlit)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
# 等待任意一个进程退出
wait
