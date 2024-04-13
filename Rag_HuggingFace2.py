#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import openai
#occasionally there are issues with the key using llamaindex and Ragas so both methods are set for the key. 
OPENAI_API_KEY="sk-xxxxxxxx"
openai.api_key=OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"

from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader("input directory")
documents = loader.load()


# In[8]:


for document in documents:
    document.metadata['filename'] = document.metadata['source']


# In[9]:


from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 


# Assuming you have already set up the necessary API keys or authentication methods for OpenAI
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model='gpt-4')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create a TestsetGenerator instance with the specified generator, critic, and embeddings
generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings
)


# Generate the test set with specified distribution among evolution types
testset = generator.generate_with_langchain_docs(
    documents=documents, 
    test_size=12, 
    raise_exceptions=False,
    distributions={
        simple: 0.5, 
        reasoning: 0.25, 
        multi_context: 0.25
    }
)



# In[10]:


testdf = testset.to_pandas()



# 

# In[11]:


from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader, Response,
                               )
import pandas as pd
import nest_asyncio

nest_asyncio.apply()
pd.set_option("display.max_colwidth", 0)
# Initialize the SimpleDirectoryReader with directory path
reader = SimpleDirectoryReader(input_dir="input_directory")

# Load the documents
documents = reader.load_data()


# Create the index from the documents
index = VectorStoreIndex.from_documents(documents)

print(f"Loaded {len(documents)} docs")

for document in documents:
    # Check if 'file_name' key exists in the metadata dictionary
    if "file_name" in document.metadata:
        # Change 'file_name' to 'filename'
        document.metadata["filename"] = document.metadata.pop("file_name")

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# list of queries
query_list = testdf['question'].tolist()
    


# dictionary to store queries and responses
query_response_dict = {}

for query in query_list:
    response = query_engine.query(query)
    query_response_dict[query] = response

for query, response in query_response_dict.items():
    print(f'Query: {query}\nResponse: {response}\n')




# In[12]:


query_response_tuples = [(query, response) for query, response in query_response_dict.items()]
responses_df = pd.DataFrame(query_response_tuples, columns=["question", 'answer'])


# In[13]:


result = testdf.merge(responses_df, on='question', how='left')


# In[15]:


#turns the resulting joined dataframe into a csv.
result.to_csv("/yourDirectory", index=False)


# In[14]:




