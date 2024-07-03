import os
import json 

from pathlib import Path  
from os import path
from glob import glob  
import shutil

from typing import List, Optional
from operator import itemgetter

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough, RunnableLambda,RunnableBranch
from langchain.schema import format_document, Document
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

from langchain.schema.runnable import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
    RunnableMap,
)

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler

from typing import Dict, Optional
import chainlit as cl


@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user

import chainlit as cl

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Ayuda",
            message="¿Qué puedes hacer por mi?",
            icon="/public/help.svg",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/idea.svg",
            ),
        ]


# Load the variables from .env
load_dotenv()

FILES_FOLDER = './doc'

# Base templates for vector store prompts
template_single_line = PromptTemplate.from_template(
    """Answer the question in a single line based on the following context.
    If there is not relevant information in the context, just say that you do not know:
{context}

Question: {question}
"""
)

template_detailed = PromptTemplate.from_template(
    """Answer the question in a detailed way with an idea per bullet point based on the following context.
    If there is not relevant information in the context, just say that you do not know:
{context}

Question: {question}
"""
)

prompt_alternatives = {
    "detailed": template_detailed,
}


configurable_prompt = template_single_line.configurable_alternatives(
    which=ConfigurableField(
        id="output_type",
        name="Output type",
        description="The type for the output, single line or detailed.",
    ),
    default_key="single_line",
    **prompt_alternatives,
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

@cl.step(type="tool")
def retrieve_documents(input):
    db = cl.user_session.get("db")
    if db:
        return db.as_retriever(search_type="similarity",search_kwargs={'k': 4}).with_config(run_name="Document retriever")
    return []


def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""

    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)

ACTIONS_QUESTION_PROMPT = PromptTemplate.from_template(
    """There are actions available for the user to modify the behaviour of the interface. Given the following question and the following action descriptions,
        please answer with the key of the action the user want to execute:
        Question: {question}
        Available actions:
            - uploadfiles: Upload files
            - reindex: Reindex or rebuild the database
            - listfiles: List the files of the knowledge database
        If there is no action please answer None
    """
)

def find_ext(dr, ext):
    return glob(path.join(dr,"*.{}".format(ext)))

async def nop (input):
    print("Nop: ", input)
    return input

async def action_response (input):
    print("Action response: ", input.content)
    return input.content

@cl.action_callback("remove_file")
async def on_action(action):
    try:
        os.remove(action.value)
        await cl.Message(content=f"Executed {action}").send()
        # Optionally remove the action button from the chatbot user interface
        await action.remove()
    except FileNotFoundError as err:
        print(f"Unexpected {err=}, {type(err)=}")


@cl.step(type="tool")
async def list_files(input):
    current_step = cl.context.current_step

    pdf_files = find_ext(FILES_FOLDER,"pdf")

    for pdf in pdf_files:
        actions = []
        actions.append( cl.Action(name="remove_file", label="Eliminar", value=pdf, description="Remove file") )
        await cl.Message(content=Path(pdf).name, actions=actions).send()

    return ""

@cl.step(type="tool")
async def reindex(input):

        app_user = cl.user_session.get("user")

        embedding = AzureOpenAIEmbeddings(
           azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
           openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        DB_FAISS_PATH = f'vectorstore/{app_user.identifier}'

        pdf_files = find_ext(FILES_FOLDER,"pdf")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        langchain_documents = []
        for document in pdf_files:
            try:
                loader = PyMuPDFLoader(document)
                data = loader.load()
                langchain_documents.extend(data)
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                continue
            
        if len(langchain_documents) > 0:
            splits = text_splitter.split_documents(langchain_documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)

            shutil.rmtree(DB_FAISS_PATH)

            vectorstore.save_local(DB_FAISS_PATH)

            await cl.Message(content="Se ha reconstruido la base de datos").send()

        return ""

@cl.step(type="tool")
async def upload_actions(input):

    current_step = cl.context.current_step

    # Override the input of the step
    print("Step input:",current_step.input)

    # Wait for the user to upload a file
    files = None

    while files == None:
        files = await cl.AskFileMessage(
                content="Please upload a text file to begin!", accept=["application/pdf"],
                max_files=10,
                max_size_mb=10
        ).send()

    for file in files:
        print(file,"-->",f'{FILES_FOLDER}/{file.name}')
        shutil.copy(file.path,f'{FILES_FOLDER}/{file.name}')

    return f"`Los archivos {files}` se han añadido a la base de datos"



def format_chat_history(chat_history: dict) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        actor = "Human" if dialogue_turn["role"] == "user" else "Assistant"
        buffer += f"{actor}: {dialogue_turn['content']}\n"
    return buffer


output_type = "detailed"


# to create a new file named vectorstore in your current directory.
def load_knowledgeBase():

        app_user = cl.user_session.get("user")
        print("User:", app_user)

        embedding = AzureOpenAIEmbeddings(
           azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
           openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        DB_FAISS_PATH = f'vectorstore/{app_user.identifier}'

        db = None

        try:
            db = FAISS.load_local(DB_FAISS_PATH, embedding,allow_dangerous_deserialization=True)
            cl.user_session.set("db",db)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

        return db


class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs
            self.sources_path = set()

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set
                self.sources_path.add(d.metadata['source'])

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            print("on_llm_end")
            if len(self.sources):
                sources_text = "\n".join([f"{Path(source).name}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

            for path in self.sources_path:
                self.msg.elements.append(
                     cl.File(
                            name=Path(path).name,
                            path=path,
                            display="inline")
                )

@cl.on_chat_start
async def on_chat_start():

    cl.user_session.set(
        "chat_history",
        [{"role": "system", "content": "You are a helpful assistant on staff regulations on University of Navarra. You must be kind and answer the questions from University employees regarding several subjects."}],
    )

    db = load_knowledgeBase()

    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        temperature=0.5, streaming=True
    )

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    actions_chain = RunnableMap(
        action = RunnablePassthrough()
        | ACTIONS_QUESTION_PROMPT
        | model
        | action_response
    )


    upload_chain =   RunnableLambda(upload_actions)

    list_chain =   RunnableLambda(list_files)

    inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | model
        | StrOutputParser(),
    )

    context = {
        "context": itemgetter("standalone_question")
                    | RunnableLambda(retrieve_documents)
                    | combine_documents,
        "question": itemgetter("standalone_question"),
    }

    chain =  inputs | context | configurable_prompt | model | StrOutputParser()

    branch = RunnableBranch(
        (lambda x: "uploadfiles" == x["action"]["action"].lower(), upload_chain),
        (lambda x: "listfiles" == x["action"]["action"].lower(), list_chain),
        (lambda x: "reindex" == x["action"]["action"].lower(), reindex),
        chain,
    )

    full_chain = {"action": actions_chain, "question": lambda x: x["question"],"chat_history": lambda x:x['chat_history']} | branch

    cl.user_session.set("question_chain", full_chain)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("chat_history")
    message_history.append({"role": "user", "content": message.content})

    chain = cl.user_session.get("question_chain")  # type: Runnable


    msg = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": message.content,"chat_history": message_history}, 
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)],
                                configurable={
                                    "output_type": output_type,
                                }),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()