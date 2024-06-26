import os
from pathlib import Path  

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


output_type = "detailed"


# to create a new file named vectorstore in your current directory.
def load_knowledgeBase():

        embedding = AzureOpenAIEmbeddings(
           azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"],
           openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embedding,allow_dangerous_deserialization=True)
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
        [{"role": "system", "content": "You are a helpful assistant on teachers regulations on University of Navarra"}],
    )

    db = load_knowledgeBase()

    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        temperature=0, streaming=True
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
                    | db.as_retriever(search_type="similarity",search_kwargs={'k': 4}).with_config(run_name="Document retriever")
                    | combine_documents,
        "question": itemgetter("standalone_question"),
    }

    chain = inputs | context | configurable_prompt | model | StrOutputParser()

    cl.user_session.set("runnable", chain)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("chat_history")
    message_history.append({"role": "user", "content": message.content})

    chain = cl.user_session.get("runnable")  # type: Runnable

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