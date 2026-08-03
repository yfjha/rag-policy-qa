import os
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from openai import OpenAI

# ====== 配置环境 ======
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com/v1")

# ====== 加载全部文档 ======
def load_all_documents():
    all_docs = []
    os.makedirs("data", exist_ok=True)
    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            filepath = os.path.join("data", filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename.replace(".txt", "")
            all_docs.extend(docs)          # 注意：extend 在 for 外面
            print(f"  加载文档: {filename}")
    print(f"共加载 {len(all_docs)} 个文档块")
    return all_docs

# 调用函数（记得加括号！）
docs = load_all_documents()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", "、"]
)
chunks = splitter.split_documents(docs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)

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

# ====== 8 个测试问题 ======
QUESTIONS = [
    "我入职2年，年假多少天？",
    "迟到怎么扣钱？",
    "出差住宿费能报销多少？",
    "P3级别基本工资多少？",
    "我周末加班工资怎么算？",
    "忘记打卡了怎么办？",
    "出差一天吃饭补贴多少？",
    "公司帮我交社保吗？",
]

# ====== Multi-Query：改写问题 ======
def generate_queries(question):
    """用 LLM 把一个问题改写成 3 个不同问法"""
    prompt = f"""你是助手，把下面问题改写成3个不同问法，保持原意，用词不同。
每行一个，不要编号。

用户问题：{question}
改写："""
    try:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        lines = r.choices[0].message.content.strip().split("\n")
        queries = [question] + [line.strip() for line in lines if line.strip()]
        print(f"  [改写] {question[:15]}... → {queries}")  
        return queries
    except Exception as e:
        print(f"改写失败，用原问题: {e}")
        return [question]

# ====== Multi-Query：多问法检索 + 合并去重 ======
def multi_query_retrieve(question, retriever):
    """多个问法分别检索，合并去重后返回"""
    queries = generate_queries(question)
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q)[:2])   # 每个问法都检索

    # 按内容去重
    seen = set()
    unique = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique

# ====== 生成回答 ======
def generate_answer(retriever, question):
    docs = multi_query_retrieve(question, retriever)   # ← Multi-Query 版
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = f"""你是一个专业的人力资源政策助手。请基于以下政策文档回答用户的问题。

【政策文档】
{context}

【用户问题】
{question}

【回答要求】
1. 准确引用政策原文
2. 如果政策中没有相关信息，请诚实告知
3. 回答要简洁、清晰

回答："""

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return r.choices[0].message.content, context, docs

# ====== 答案关键词表（Hit Rate 用） ======
HIT_KEYWORDS = {
    "我入职2年，年假多少天？": ["5天"],
    "迟到怎么扣钱？": ["50元"],
    "出差住宿费能报销多少？": ["500", "350"],
    "P3级别基本工资多少？": ["8000"],
    "我周末加班工资怎么算？": ["2倍"],
    "忘记打卡了怎么办？": ["补卡"],
    "出差一天吃饭补贴多少？": ["150", "100"],
    "公司帮我交社保吗？": ["五险一金"],
}

def eval_hit_rate(question, docs):
    """检索命中率：关键词匹配，不依赖 LLM"""
    keywords = HIT_KEYWORDS.get(question, [])
    for doc in docs:
        for kw in keywords:
            if kw in doc.page_content:
                return 1.0
    return 0.0

# ====== 打分函数 ======
def eval_faithfulness(answer, context):
    prompt = f"""判断回答是否严格基于文档内容生成。
文档：{context}
回答：{answer}
只输出数字（0.0/0.5/1.0），不要其他文字。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return float(r.choices[0].message.content.strip())

def eval_relevancy(question, answer):
    prompt = f"""判断回答是否直接回答了用户的问题。
问题：{question}
回答：{answer}
只输出数字（0.0/0.5/1.0），不要其他文字。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return float(r.choices[0].message.content.strip())

# ====== 对比实验 ======
K_VALUES = [4]

print("=" * 70)
print("K 值对比实验报告（Multi-Query 版）")
print("=" * 70)

for k in K_VALUES:
    print(f"\n{'=' * 70}")
    print(f"  Top-K = {k}（Multi-Query）")
    print(f"{'=' * 70}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    total_f = total_r = total_hit = 0

    for i, q in enumerate(QUESTIONS):
        answer, context, docs = generate_answer(retriever, q)
        hit = eval_hit_rate(q, docs)
        f = eval_faithfulness(answer, context)
        r = eval_relevancy(q, answer)
        total_f += f
        total_r += r
        total_hit += hit
        print(f"  Q{i+1}: {q[:20]}...  Faith={f:.1f}  Rel={r:.1f}  Hit={hit:.1f}")

    avg_f = total_f / len(QUESTIONS)
    avg_r = total_r / len(QUESTIONS)
    avg_hit = total_hit / len(QUESTIONS)
    print(f"  >> 平均: Faithfulness={avg_f:.2f}  Relevance={avg_r:.2f}  Hit Rate={avg_hit:.2f}")

print("\n" + "=" * 70)
print("对比（普通 vs Multi-Query 的 Hit Rate）：")
print("  普通 K=1: 0.62   K=4: 0.88   K=8: 0.88")
print("  现在跑 Multi-Query 版，看能不能提到 1.00！")
print("=" * 70)
   