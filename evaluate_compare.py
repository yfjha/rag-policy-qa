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

MODEL_NAME ="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com/v1")

# ====== 初始化文档和 Embedding ======
def load_all_documents():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # 给每个文档块加上来源文件名
            for doc in docs:
                doc.metadata["source"] = filename.replace(".txt", "")
            all_docs.extend(docs)
            print(f"  加载文档: {filename}")
    
    print(f"共加载 {len(all_docs)} 个文档块")
    return all_docs

# 调用加载函数
docs = load_all_documents()



spilitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", "、"]    
)
chunks = spilitter.split_documents(docs)

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
vector_store = FAISS.from_documents(chunks,embeddings)

# ====== 测试问题（8 个：4 直接 + 4 措辞不匹配） ======
QUESTIONS = [
    # 文档里有原词
    "我入职2年，年假多少天？",
    "迟到怎么扣钱？",
    "出差住宿费能报销多少？",
    "P3级别基本工资多少？",
    # 文档里有答案，但说法不一样
    "我周末加班工资怎么算？",
    "忘记打卡了怎么办？",
    "出差一天吃饭补贴多少？",
    "公司帮我交社保吗？",
]

def generate_answer(retriever,question):
    docs = retriever.invoke(question)
    context ="\n\n---\n\n".join([d.page_content for d in docs])

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
    return r.choices[0].message.content,context,docs

# ====== 检测实验 ======
def eval_faithfulness(answer,context):
    """忠实度评分"""
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
    """相关性评分"""
    prompt = f"""判断回答是否直接回答了用户的问题。
问题：{question}
回答：{answer}
只输出数字（0.0/0.5/1.0），不要其他文字。"""
    r = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0)
    return float(r.choices[0].message.content.strip())


# ====== 答案关键词表（每个问题的答案长什么样） ======
HIT_KEYWORDS = {
    "我入职2年，年假多少天？": ["5天"],          # 答案里有 5天
    "迟到怎么扣钱？":           ["50元"],
    "出差住宿费能报销多少？":   ["500", "350"],   # 关键词里放数字
    "P3级别基本工资多少？":     ["8000"],
    "我周末加班工资怎么算？":   ["2倍"],
    "忘记打卡了怎么办？":       ["补卡"],
    "出差一天吃饭补贴多少？":   ["150", "100"],
    "公司帮我交社保吗？":       ["五险一金"],
}

def eval_hit_rate(question,docs):
    """
    检索命中率：
    检查检索到的文档块里是否包含答案关键词
    返回 1.0（命中）或 0.0（未命中）
    """
    keywords = HIT_KEYWORDS.get(question,[])
    for doc in docs:
        for kw in keywords:
            if kw in doc.page_content:
                return 1.0
    return 0.0
        

# ====== 对比实验 ======
K_VALUES =[1,4,8]

print("=" * 70)
print("K 值对比实验报告")
print("=" * 70)

for k in K_VALUES:
    print(f"\n{'=' * 70}")
    print(f"  Top-K = {k}")
    print(f"{'=' * 70}")

    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": k}
    )
    total_f = total_r =total_hit=0

    for i,q in enumerate(QUESTIONS):
        answer,context,docs = generate_answer(retriever,q)
        hit = eval_hit_rate(q,docs)
        f = eval_faithfulness(answer, context)
        r = eval_relevancy(q, answer)
        total_f += f
        total_r += r
        total_hit +=hit
        print(f"  Q{i+1}: {q[:20]}...  Faith={f:.1f}  Relevance={r:.1f}")


    avg_f = total_f / len(QUESTIONS)
    avg_r = total_r / len(QUESTIONS)
    avg_hit = total_hit / len(QUESTIONS)
    print(f"  >> 平均: Faithfulness={avg_f:.2f}  Relevance={avg_r:.2f} Hit Rate={avg_hit:.2f}  "f"综合={(avg_f+avg_r)/2:.2f}")


print(f"\n{'=' * 70}")
print("实验结论：")
print(f"{'=' * 70}")
print("K 值越大 → 检索到的文档块越多 → 可能提高相关性")
print("        但过多的文档块可能引入噪声 → 降低忠实度")
print("最佳 K 值需要根据实际评估数据选择")
print("=" * 70)     