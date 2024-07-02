import os
from tqdm import tqdm

from os import path
from glob import glob  

def find_ext(dr, ext):
    return glob(path.join(dr,"*.{}".format(ext)))

from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
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

        pdf_files = find_ext("./doc","pdf")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        langchain_documents = []
        for document in tqdm(pdf_files):
            try:
                loader = PyMuPDFLoader(document)
                data = loader.load()
                langchain_documents.extend(data)
            except Exception:
                continue

        splits = text_splitter.split_documents(langchain_documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)

        vectorstore.save_local(DB_FAISS_PATH)