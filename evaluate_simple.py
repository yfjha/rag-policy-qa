# evaluate_simple.py - 用 DeepSeek 直接评估 RAG 系统
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com/v1")

# 测试数据：question(问题), answer(系统回答), context(检索到的文档), reference(标准答案)
test_cases = [
    {
        "question": "我入职2年，年假多少天？",
        "answer": "根据公司年假政策（2026年）第一条第1款，入职满1年但不满3年的员工每年享有5天带薪年假。您入职2年，属于该区间，因此您的年假为5天。",
        "context": "一、年假计算标准\n1. 入职满1年但不满3年：每年5天\n2. 入职满3年但不满5年：每年10天\n3. 入职满5年及以上：每年15天",
        "reference": "入职2年，每年5天带薪年假。"
    },
    {
        "question": "年假可以顺延到下一年吗？",
        "answer": "根据公司年假政策（2026年）第二条第2款，年假最多顺延5天至下一年度，顺延的年假需在次年3月31日前使用完毕。",
        "context": "二、年假使用规则\n2. 年假最多顺延5天至下一年度\n3. 顺延的年假需在次年3月31日前使用完毕",
        "reference": "年假最多顺延5天至下一年度。"
    },
    {
        "question": "离职时未休完的年假怎么补偿？",
        "answer": "根据公司年假政策（2026年）第三条第1款，离职时未休完的年假按日工资3倍补偿。日工资=月基本工资/21.75天。",
        "context": "三、离职处理\n1. 离职时未休完的年假按日工资3倍补偿\n2. 日工资=月基本工资/21.75天",
        "reference": "未休年假按日工资3倍补偿。"
    },
    {
        "question": "入职3年可以休几天年假？",
        "answer": "根据公司年假政策（2026年）第一条第2款，入职满3年但不满5年的员工每年享有10天带薪年假。",
        "context": "一、年假计算标准\n2. 入职满3年但不满5年：每年10天\n3. 入职满5年及以上：每年15天",
        "reference": "入职3年，每年10天带薪年假。"
    },
    {
        "question": "年假怎么申请？",
        "answer": "根据公司年假政策（2026年）第二条第1款，年假可以按半天为单位分批次使用。同时也可以申请顺延，最多顺延5天至下一年度。",
        "context": "二、年假使用规则\n1. 年假可以按半天为单位分批次使用\n2. 年假最多顺延5天至下一年度",
        "reference": "年假可按半天为单位分批次使用。"
    },
]

def eval_faithfulness(answer, context):
    """检查回答是否基于文档（忠实度）"""
    prompt = f"""你是一个严格的评判员。请判断以下"回答"是否严格基于"文档内容"生成，没有编造信息。

文档内容：
{context}

回答：
{answer}

请按以下标准评分（0-1分）：
- 1.0：回答完全基于文档，没有编造
- 0.5：回答大部分基于文档，但有部分推断
- 0.0：回答明显编造了文档中没有的内容

只输出一个数字（0.0/0.5/1.0），不要输出其他文字。"""
    
    r = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
    return float(r.choices[0].message.content.strip())

def eval_relevancy(question, answer):
    """检查回答是否回答了问题（相关性）"""
    prompt = f"""你是一个严格的评判员。请判断以下"回答"是否直接回答了用户的问题。

用户问题：{question}
回答：{answer}

请按以下标准评分（0-1分）：
- 1.0：准确回答了问题，完全相关
- 0.5：部分相关，但没有完全回答问题
- 0.0：没有回答问题或答非所问

只输出一个数字（0.0/0.5/1.0），不要输出其他文字。"""
    
    r = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
    return float(r.choices[0].message.content.strip())

def eval_recall(answer, reference):
    """检查回答是否覆盖了标准答案中的信息（召回率）"""
    prompt = f"""你是一个严格的评判员。请判断以下"回答"是否覆盖了"标准答案"中的所有关键信息。

标准答案：{reference}
回答：{answer}

请按以下标准评分（0-1分）：
- 1.0：完全覆盖了标准答案中的关键信息
- 0.5：覆盖了部分关键信息，有遗漏
- 0.0：没有覆盖任何关键信息

只输出一个数字（0.0/0.5/1.0），不要输出其他文字。"""
    
    r = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
    return float(r.choices[0].message.content.strip())

# 运行评估
print("=" * 60)
print("RAG 系统评估报告")
print("=" * 60)

total_f = total_r = total_rec = 0
results = []

for i, case in enumerate(test_cases):
    print(f"\n第 {i+1} 条：{case['question']}")
    print("-" * 40)
    
    f = eval_faithfulness(case["answer"], case["context"])
    r = eval_relevancy(case["question"], case["answer"])
    rec = eval_recall(case["answer"], case["reference"])
    
    total_f += f
    total_r += r
    total_rec += rec
    
    results.append((case["question"], f, r, rec))
    print(f"  忠实度(Faithfulness):     {f:.1f}")
    print(f"  相关性(Relevancy):       {r:.1f}")
    print(f"  召回率(Recall):          {rec:.1f}")

n = len(test_cases)
print("\n" + "=" * 60)
print("综合评分：")
print(f"  忠实度 (Faithfulness):  平均 {total_f/n:.2f}")
print(f"  相关性 (Relevancy):    平均 {total_r/n:.2f}")
print(f"  召回率 (Recall):       平均 {total_rec/n:.2f}")
print(f"  综合得分:             平均 {(total_f+total_r+total_rec)/(3*n):.2f}")
print("=" * 60)
