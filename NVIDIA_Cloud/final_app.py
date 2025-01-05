import streamlit  as st 
import os 
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings , ChatNVIDIA 
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS 

from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate  

from dotenv import load_dotenv 
load_dotenv() 

# print(os.getenv("NVIDIA_API_KEY"))

os.environ["NVIDIA_API_KEY"]=os.getenv("NVIDIA_API_KEY")  


def vector_embedding()  :
    
    
    if "vectors" not in st.session_state:

        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
        
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


st.title("RAG Document Query Proj with Nvidia NIM ")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct") 



prompt=ChatPromptTemplate.from_template(""" 
Answer the the Question based on the provided context only 
please provide the most accurate response based on the question 
<context>
{context}
<context> 
question:{input}

""") 


user_input=st.text_input("Enter Your Question From Doduments") 


if st.button("Documents Embedding"):
    vector_embedding()
    st.success("Vector Store DB Is Ready U Can ask Quesiton now") 
    

if user_input:
    document_chain = create_stuff_documents_chain(llm=llm , prompt=prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain) 
    
    start=time.process_time() 
    response=retrieval_chain.invoke({'input':user_input})
    print("Response time :",time.process_time()-start) 
    
    st.write(response["answer"])
    
    with st.expander("Document Similarity serach:"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")