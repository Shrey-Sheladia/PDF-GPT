import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import openai
import os
import pprint 
import chardet
pp = pprint.PrettyPrinter(indent=4)
from uuid import uuid4
try:
    from dotenv import load_dotenv
except:
    pass

load_dotenv()

# Access the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

print(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV)

print("D")

# Function to calculate token length using tiktoken
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Function to split the text using langchain's RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)
    

# Function to convert text to embeddings using OpenAI's 'text-embedding-ada-002' model
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY)



# Main function to ingest text into Pinecone index
def ingest_text(text, doc_name, cluster_name, index_name):
    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    print("DONE")

    # Check if the index already exists
    print(pinecone.list_indexes())
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    # Connect to the index
    index = pinecone.Index(index_name)

    # Split the text into chunks
    text_chunks = text_splitter.split_text(text)

    batchSize = 10

    texts, metadatas = [], []

    metadata = {
        "doc_name": doc_name,
        "cluster_name": cluster_name
    }

    chunk_metadata = [{"text": chunk, "chunk_num": i, **metadata} for i, chunk in enumerate(text_chunks)]

    texts.extend(text_chunks)
    metadatas.extend(chunk_metadata)

    if len(texts) >= batchSize:
        print("upserting")
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []
        print("upserted")
    
    if len(texts) > 0:
        print("upserting")
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        print("upserted")


def query_index(query, cluster_name, index_name, metadata=None):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    text_field = "text"
    index = pinecone.Index(index_name)

    vectorstore = Pinecone.from_existing_index(index_name, embed)
    response = vectorstore.similarity_search(
        query,  # our search query
    )
    for res in response:
        print(f"Chunk: {res.metadata['chunk_num']}")
        print(f"Content:\n{res.page_content}")
        print("\n---\n")
    
    print("_"*200)




input_file1 = 'Book 3 - The Prisoner of Azkaban.txt' 
input_file = 'textDocs/' + input_file1

with open(input_file, "rb") as f:
    encoding = chardet.detect(f.read())["encoding"]
with open(input_file, "r", encoding=encoding) as f:
    txt = f.read()
    ingest_text(txt, input_file1, "Trials", "test-index")



# question = "What is Percy's relationship with his father?"
# query_index(question, "Trials", "test-index")


system_message = '''
REPLY AS A PYTHON DICTIONARY! 
You are a chat assistant that will create answers based on the information provided. 

The message will contain the Query, followed by the context. This will state the "chunk_num" followed by the chunk content. You will only use the information provided in these chunks/messages to create your response. You do not need to use all of the information provided, just what you think is relevant to the query. Do not add unnecessary details. You can also use markdown formatting if required for a better layout.
Your response should be in the form of a Python dictionary, with one key being "reply" with your response, and the other being "source" which will have a list as the value. This list should contain all the chunk numbers from which you used information/facts to formulate your answer. 
The response should look like this:
{"Reply": "Your answer here", "Source": [0.0, 1.0]} (This is just an example)
The user should be able to read that chunk to check the response.

If the context/information provided by the user does not have enough information related to the Query, you will respond stating that.

REPLY AS A PYTHON DICTIONARY! 


'''
