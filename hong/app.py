import streamlit as st
from preprocess.send_vectordb import setterQdrant, delete_collection
from llm_custom import rag_ans

def main():
    st.title("WOORIFISA RAG prototype")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=False)
        process = st.button("Process")
        delete_collect = st.button("delete_collection")
        rag_func = st.radio(
            "RAG 적용 여부를 선택해주세요",
            ["O", "X"],
            captions=[
                "RAG 적용",
                "RAG 미적용"
            ],
        )

    if process:
        setterQdrant(uploaded_files)
        st.session_state.processComplete = True
    
    if delete_collect:
        delete_collection()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! WOORIFISA RAG Prototype 테스트를 할 수 있습니다."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if rag_func == 'O':
                    result = rag_ans(True, query)
                else:
                    result = rag_ans(False, query)
                response = result

                st.markdown(response)
                    
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()