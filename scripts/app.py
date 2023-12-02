"""

Multiple PDF Chat Application

This Streamlit-based application allows users to upload multiple PDF documents and engage in a conversation based on the content of these documents using OpenAI's language model.

The application performs the following tasks:
- Extracts text content from uploaded PDFs.
- Divides the text into manageable chunks.
- Creates a vector store for language model interaction.
- Establishes a conversational chain for user interaction.

Author: [MEF]

"""


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os


def get_pdf_text(pdf_docs):
    
    """
    Extracts text content from the provided PDF documents.

    This function takes a list of PDF documents as input and reads each document using PyPDF2's PdfReader. It iterates through each page of every PDF and extracts text content, concatenating it into a single string.

    Parameters:
    - pdf_docs (list): A list containing paths or file-like objects of PDF documents.

    Returns:
    - text (str): The concatenated text content from all the provided PDF documents.
    """
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    
    """
    Splits the input text into smaller chunks for processing.

    This function takes a large text input and divides it into smaller, manageable chunks. It uses a `CharacterTextSplitter` object with specified parameters like separator, chunk size, overlap, and length function to split the text.

    Parameters:
    - text (str): The input text to be divided into smaller chunks.

    Returns:
    - chunks (list): A list containing smaller chunks of text, divided based on the specified parameters.
    """
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    
    """
    Generates a vector store from provided text chunks using embeddings.

    This function creates a vector store from a list of text chunks by leveraging OpenAI's embeddings. It initializes an OpenAIEmbeddings object using the specified API key and then creates a FAISS vector store by processing the provided text chunks through these embeddings.

    Parameters:
    - text_chunks (list): A list containing smaller chunks of text to generate vectors.

    Returns:
    - vectorstore: A FAISS vector store created from the text chunks processed through the embeddings.
    """
    
    OPENAI_API_KEY = ""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    
    """
    Constructs a conversational chain using vector store and language models.

    This function sets up a conversational chain by initializing a ChatOpenAI language model (`llm`), creating a ConversationBufferMemory (`memory`) to store conversation history, and constructing a ConversationalRetrievalChain. The chain is built using the provided vector store as a retriever for responses.

    Parameters:
    - vectorstore: The vector store containing text representations.

    Returns:
    - conversation_chain: A ConversationalRetrievalChain object for conducting conversations based on the vector store and language model.
    """
    
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    
    """
    Handles user input for conducting a conversation.

    This function takes a user question as input, interacts with the ongoing conversation maintained in the session state, and retrieves a response from the conversational chain. It updates the chat history in the session state and displays the conversation messages using appropriate templates for user and bot messages.

    Parameters:
    - user_question (str): The question asked by the user.

    Action:
    - Retrieves a response from the conversation chain based on the user question.
    - Updates the chat history in the session state.
    - Displays the conversation messages using templates for user and bot messages.

    Returns:
    - None
    """
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    
    """
    Primary function orchestrating the PDF-based chat application.

    This function serves as the entry point for the application. It configures the Streamlit page, manages session state variables for conversation and chat history, and provides an interface for users to interact by asking questions related to uploaded PDF documents.

    Action:
    - Sets the OpenAI API key in the environment variables.
    - Configures Streamlit page settings for the application.
    - Checks and initializes session state variables for conversation and chat history.
    - Displays the application header and provides a text input for user questions.
    - Handles user input, triggering conversation handling with 'handle_userinput' function.
    - Provides a sidebar interface for users to upload PDF documents and process them.
    - Processes uploaded PDFs by extracting text, chunking it, and creating a vector store and conversation chain.

    Returns:
    - None
    """
    
    os.environ['OPENAI_API_KEY'] = ''

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()