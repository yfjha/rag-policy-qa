# fastapi_rag.py
import os
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings 
from langchain_core.embeddings import Embeddings
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi.responses import StreamingResponse   # 用于返回流式响应
from typing import AsyncGenerator                 # 类型注解（可选）

# ====== 强制离线 ======
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# ====== 配置 ======
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ====== 初始化 RAG 系统 ======
print("=" * 60)
print("正在启动 RAG 服务...")
print("=" * 60)

def init_policy():
    os.makedirs("data", exist_ok=True)
    policy_file = "data/policy.txt"
    if not os.path.exists(policy_file):
        with open(policy_file, "w", encoding="utf-8") as f:
            f.write("""公司年假政策（2026年）：

一、年假计算标准
1. 入职满1年但不满3年：每年5天带薪年假
2. 入职满3年但不满5年：每年10天带薪年假
3. 入职满5年及以上：每年15天带薪年假
4. 新员工入职当年，按实际工作月份折算（满1个月算1天）

二、年假使用规则
1. 年假可以按半天为单位分批次使用
2. 年假最多顺延5天至下一年度
3. 顺延的年假需在次年3月31日前使用完毕

三、离职处理
1. 离职时未休完的年假按日工资3倍补偿
2. 日工资 = 月基本工资 / 21.75天
""")
    return policy_file

# ====== 全局对话记录 ======
chat_history = []

def build_rag_chain():
    policy_file = init_policy()
    
    loader = TextLoader(policy_file, encoding="utf-8")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", "、"]
    )
    chunks = splitter.split_documents(docs)
    print(f"文档已分割为 {len(chunks)} 个片段")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        MODEL_NAME, 
        device=device,
        local_files_only=True
    )
    print(f"Embedding模型加载完成 (设备: {device})")
    
    class MyEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model
            self.dimension = model.get_sentence_embedding_dimension()
        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text):
            return self.model.encode(text, normalize_embeddings=True).tolist()
    
    embeddings = MyEmbeddings(model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"向量库构建完成，共 {vector_store.index.ntotal} 个向量")
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com/v1",
        timeout=60
    )
    print("LLM初始化完成")
    
    template = """你是一个专业的人力资源政策助手。请基于以下政策文档回答用户的问题。

【政策文档】
{context}

【用户问题】
{question}

【回答要求】
1. 准确引用政策原文
2. 如果政策中没有相关信息，请诚实告知
3. 回答要简洁、清晰

回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    print("RAG链创建完成")
    return rag_chain

# 构建 RAG 链
rag_chain = build_rag_chain()
print("RAG 服务初始化完成！")
print("=" * 60)

# ====== FastAPI 服务 ======
app = FastAPI(
    title="HR政策问答系统",
    description="基于RAG的年假政策智能问答系统",
    version="1.0.0"
)

# 请求/响应模型
class Question(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "我入职2年,年假多少天?"
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

# ====== API 接口 ======
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "HR政策问答系统",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - 提问接口",
            "/ask/stream": "POST - 流式问答(SSE)",
            "/docs": "GET - API文档",
            "/health": "GET - 健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "rag_chain": "loaded",
        "model": MODEL_NAME
    }

@app.post("/ask/stream")
async def ask_stream(question: Question):
    """
    流式问答接口
    返回 SSE 格式的流式响应，逐字输出答案
    """
    async def generate():
        full_answer=""
        try:
            if chat_history:
                history_text = "\n".join([
                    f"用户：{h}\n助手：{a}" for h, a in chat_history[-5:]
                ])
                enhanced_query = f"""之前的对话：
{history_text}
现在用户的新问题是：{question.query}"""
            else:
                enhanced_query = question.query                

            async for chunk in rag_chain.astream(enhanced_query):
                content = chunk.content
                if content:
                    full_answer+=content
                    yield f"data: {content}\n\n"
            # 保存到历史记录
            chat_history.append((question.query, full_answer))
            # 限制长度，最多保留 10 轮
            if len(chat_history) > 10:
                chat_history.pop(0)            
            # 结束标志
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 防止 nginx 缓冲
        }
    )



# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status="error"
        ).dict()
    )

# ====== 启动服务 ======
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("启动 FastAPI 服务...")
    print("=" * 60)
    print("API文档: http://localhost:8000/docs")
    print("交互式文档: http://localhost:8000/redoc")
    print("健康检查: http://localhost:8000/health")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
