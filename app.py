import streamlit as st
import requests

st.set_page_config(page_title="HR 政策问答", page_icon="🤖")
st.title("HR 年假政策问答系统 🤖")

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调 RAG API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            resp = requests.post(
                "http://localhost:8000/ask/stream",
                json={"query": prompt},
                stream=True,
                timeout=30
            )
            for line in resp.iter_lines():
                if line:
                    text = line.decode("utf-8").replace("data: ", "")
                    if "[DONE]" in text:
                        break
                    if "[ERROR]" in text:
                        full_response += text
                        break
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.markdown(f"连接失败：{str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
