from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from Utils import *


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        if 'knowledge_base' not in st.session_state or str(st.session_state.pdf) != str(pdf):
            print(str(st.session_state.pdf) == str(pdf) if "pdf" in st.session_state else print("No PDF"))
            print(st.session_state.pdf,pdf, sep = "\n")
            print(st.session_state)
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
    

if __name__ == '__main__':
    main()
