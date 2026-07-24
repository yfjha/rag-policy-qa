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
os.makedirs("data",exist_ok=True)
policy_file = "data/policy.txt"
if not os.path.exists(policy_file):
    with open(policy_file,"w",encoding="utf-8") as f:
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

loader = TextLoader(policy_file,encoding="utf-8")
docs = loader.load()


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

# ====== 测试问题（5 个） ======
QUESTIONS = [
    "我入职2年，年假多少天？",
    "年假可以顺延到下一年吗？",
    "离职时未休完的年假怎么补偿？",
    "入职3年可以休几天年假？",
    "年假怎么申请？",
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
    return r.choices[0].message.content,context

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
    r = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
    return float(r.choices[0].message.content.strip())

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
    total_f = total_r =0

    for i,q in enumerate(QUESTIONS):
        answer,context = generate_answer(retriever,q)
        f = eval_faithfulness(answer, context)
        r = eval_relevancy(q, answer)
        total_f += f
        total_r += r
        print(f"  Q{i+1}: {q[:20]}...  Faith={f:.1f}  Relevance={r:.1f}")


    avg_f = total_f / len(QUESTIONS)
    avg_r = total_r / len(QUESTIONS)
    print(f"  >> 平均: Faithfulness={avg_f:.2f}  Relevance={avg_r:.2f}  "f"综合={(avg_f+avg_r)/2:.2f}")


print(f"\n{'=' * 70}")
print("实验结论：")
print(f"{'=' * 70}")
print("K 值越大 → 检索到的文档块越多 → 可能提高相关性")
print("        但过多的文档块可能引入噪声 → 降低忠实度")
print("最佳 K 值需要根据实际评估数据选择")
print("=" * 70)     