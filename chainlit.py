import os
from typing import List, Optional
from operator import itemgetter

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain.schema import format_document, Document

from langchain.schema.runnable import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
    RunnableMap,
)

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


import chainlit as cl

# Load the variables from .env
load_dotenv()

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


def format_chat_history(chat_history: dict) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        actor = "Human" if dialogue_turn["role"] == "user" else "Assistant"
        buffer += f"{actor}: {dialogue_turn['content']}\n"
    return buffer


vector_store_topic = None
output_type = None


# to create a new file named vectorstore in your current directory.
def load_knowledgeBase():

        embedding = AzureOpenAIEmbeddings(
           azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
           openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embedding,allow_dangerous_deserialization=True)
        return db

@cl.on_chat_start
async def on_chat_start():

    db = load_knowledgeBase()

    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    )

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

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
                    | db.as_retriever(search_kwargs={'k': 3})
                    | combine_documents,
        "question": itemgetter("standalone_question"),
    }

    chain = inputs | context | configurable_prompt | model | StrOutputParser()

    cl.user_session.set("runnable", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": message.content,"chat_history": []},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()