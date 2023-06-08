import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import openai
import os
import time
import sys
import pprint 
import chardet
pp = pprint.PrettyPrinter(indent=4)
from uuid import uuid4
from sentence_transformers import SentenceTransformer
import json

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
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY)


def getIds(texts):
    ids = [str(uuid4()) for _ in range(len(texts))]
    return ids

def get_embeddings(texts, open_source=True):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if open_source:
        return model.encode(texts)

    return embed.embed_documents(texts)


def get_query_embeddings(texts, open_source=True):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if open_source:
        return [float(x) for x in model.encode(texts)]

    return embed.embed_documents(texts)

def split_texts(text):
    record_texts = text_splitter.split_text(text)

    return record_texts


def upsert_text(doc_name, cluster_name, index_name):

    # input_file = "largeText.txt"

    input_file = doc_name

    with open(input_file, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open(input_file, "r", encoding=encoding) as f:
        text = f.read()

    batch_limit = 100
    batch_texts, batch_metadatas = [], []


    # doc_name, cluster_name, index_name = input_file1, "Trials", "open-source-index"

    if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                metric='dotproduct',
                dimension=1536  # 1536 dim of text-embedding-ada-002
            )
    index = pinecone.Index(index_name)

    doc_metadata = {
            "doc_name": doc_name,
            "cluster_name": cluster_name
        }
    
    all_texts = split_texts(text)
    record_metadatas = []

    BATCHES = []
    SingleBatch = []
    Batch_Embedding = []

    for j, text in enumerate(all_texts):
        print(j, end = ", ")

        # print(all_texts)

        chunk_metadata = {"chunk": j, "text": text, **doc_metadata}

        SingleBatch.append({
            "id" : str(uuid4()),
            "metadata": chunk_metadata
        })

        batch_texts.append(text)
        batch_metadatas.append(chunk_metadata)

        if len(batch_texts) >= batch_limit:
            print("\nLimit\n")
            embeddings = get_embeddings(batch_texts)

            for data, em in zip(SingleBatch, embeddings):
                EmList = []
                for em in list(em):
                    EmList.append(float(em))
                data["values"] = list(EmList)

            BATCHES.append(SingleBatch)

            SingleBatch = []
            batch_texts, batch_metadatas = [], []


    if len(batch_texts):
        print("\nLimit Last\n")

        embeddings = get_embeddings(batch_texts)

        for data, em in zip(SingleBatch, embeddings):
            EmList = []
            for em in list(em):
                EmList.append(float(em))
            data["values"] = list(EmList)

        BATCHES.append(SingleBatch)

        SingleBatch = []
        batch_texts, batch_metadatas = [], []


    for batch in BATCHES:
        print(batch)
        index.upsert(vectors=batch)


def create_gpt_question(query, Responses):

    FinalMessage = f"Query: '{query}'\n\nContexts:\n\n\n"

    for resp in Responses:
        FinalMessage += f"Chunk: {resp['Source']}\n"
        FinalMessage += f"Content: {resp['Text']}\n\n"
        FinalMessage += f"---\n\n\n"

    
    return FinalMessage




def ask_gpt(UserMessage, messages=None):

    SYSTEM_MESSAGE = '''
    REPLY AS A PYTHON DICTIONARY! 
    You are a chat assistant that will create answers based on the information provided. 

    The message will contain the Query, followed by the context. This will state the "chunk_num" followed by the chunk content. You will only use the information provided in these chunks/messages to create your response. You do not need to use all of the information provided, just what you think is relevant to the query. Do not add unnecessary details. You can also use markdown formatting if required for a better layout.
    Your response should be in the form of a Python dictionary, with one key being "reply" with your response, and the other being "source" which will have a list as the value. This list should contain all the chunk numbers from which you used information/facts to formulate your answer. 
    The response should look like this:
    {"Reply": "Your answer here", "Source": [0.0, 1.0]} (This is just an example)
    The user should be able to read that chunk to check the response.

    If the context/information provided by the user does not have enough information related to the Query, you will respond stating that.

    REPLY AS A PYTHON DICTIONARY!  USE MARKDOWN IF REQUIRED!


    '''

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": UserMessage}
    ]
    
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,)
    response_text = completion["choices"][0]["message"]["content"]

    


    try:
        response_dict = json.loads(response_text)
        Reply = response_dict.get('Reply', None)
        Source = response_dict.get('Source', None)
    except json.JSONDecodeError:
        print("JSON Decode Error")
        Reply = response_text
        Source = None

    return Reply, Source


def query_index(query, cluster_name, index_name, filter=None, openSource=True):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    
    text_field = "text"
    index = pinecone.Index(index_name)

    Responses = []


    
    if openSource:

        response = index.query(
        top_k=3,
        include_values=False,
        include_metadata=True,
        vector=get_query_embeddings(query),
        filter={}
    )
        
        for match in (response["matches"]):
            source = (match["metadata"]["chunk"])
            doc_name = (match["metadata"]["doc_name"])
            text = (match["metadata"]["text"])

            Responses.append({"Source" : source, "Doc": doc_name, "Text": text})

        return create_gpt_question(query, Responses)
          
    else:
        vectorstore = Pinecone.from_existing_index(index_name, embed)
        
        response = vectorstore.similarity_search(
            query,  # our search query
        )
        for res in response:
            print(f"Chunk: {res.metadata['chunk']}")
            print(f"Content:\n{res.page_content}")
        
    print("_"*200)



def get_answer(query):
    user_message = query_index(query, "Trials", "open-source-index")
    proper_response = ask_gpt(user_message)

    return proper_response

input_file1 = 'Book 5 - The Order of the Phoenix.txt' 
input_file = 'textDocs/' + input_file1


# input("Upsert? >")
# upsert_text(input_file, "Trials", "open-source-index")

question = ""

while question.lower() != "q":
    question = input("Ask a question> ")
    if question.lower() == "q":
        break


    print("\n\nResponse:\n")
    try:
        print(get_answer(question))
    except Exception as e:
        print(str(e))


