import streamlit as st
from Utils import *
import time

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


st.set_page_config(page_title="PDF GPT")
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.header("Talk with your PDF! 💬")

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    if 'knowledge_base' not in st.session_state or str(st.session_state.pdf) != str(pdf):

        # Create Knowledge base
        st.session_state.pdf = pdf
        st.session_state.knowledge_base = create_knowledge_base_from_pdf(pdf)
    

    answer = st.empty()
    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:

        response, cb = get_response(user_question, st.session_state.knowledge_base)
        print(cb)
            
        answer.write(response)
