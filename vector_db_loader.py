import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

#create a new file named vectorstore in your current directory.
if __name__=="__main__":
    
        embedding = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        DB_FAISS_PATH = 'vectorstore/db_faiss'
        loader=PyPDFLoader("./doc/Normativa_profesorado.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
        vectorstore.save_local(DB_FAISS_PATH)