try:
    from dotenv import load_dotenv
except:
    pass
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import faiss

load_dotenv()
def get_chunks(pdf_file):
    print("Creating Chunks")
    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    
    chunks = text_splitter.split_text(text)
    print("Created Chunks")
    return chunks
 

def create_knowledge_base_from_pdf(pdf):
    print("Creating Knowledge Base")
    chunks = get_chunks(pdf)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    print("Created Knowledge Base")
    return knowledge_base



def get_response(user_question, knowledge_base):
    print("Getting Docs")
    docs = knowledge_base.similarity_search(user_question)
    print("Got Docs")

    print("Getting Respnse")
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)

        print("Got Respnse")
        return response, cb


def merge_faiss_indices_from_pdf(pdf_path, existing_index_path):
    # Load existing index
    existing_index = faiss.read_index(existing_index_path)

    # Create new index from PDF
    chunks = get_chunks(pdf_path)
    embeddings = OpenAIEmbeddings()
    new_index = FAISS.from_texts(chunks, embeddings)

    # Merge the new index into the existing index
    combined_index = faiss.IndexFlatIP(existing_index.d + new_index.d)
    faiss.merge_into_lattice(existing_index, new_index, combined_index)

    # Save the combined index to disk
    faiss.write_index(combined_index, existing_index_path)

    # Return the combined index
    return combined_index

    





