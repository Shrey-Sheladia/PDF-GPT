import streamlit as st
from Utils import *

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

