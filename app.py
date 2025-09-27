import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# --- Load Environment Variables ---
load_dotenv(find_dotenv())

# --- Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Caching Functions to Load Models and Data ---
@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store from the local path."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Sets a custom prompt template for the QA chain."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# --- Main Streamlit App Logic ---
def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Medical Assistant",
        page_icon="🧑‍⚕️",
        layout="centered",
        initial_sidebar_state="auto"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.title("🧑‍⚕️ AI Medical Assistant")
        st.markdown("""
        This chatbot is powered by large language models and provides information based on a custom medical knowledge base. 
        
        **Disclaimer:** This is an AI assistant and not a substitute for professional medical advice. Always consult with a qualified healthcare provider for any medical concerns.
        """)
        
        st.header("⚙️ Chat Settings")
        # Add sliders for temperature and k
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                                help="Controls randomness. Lower values are more deterministic.")
        k = st.slider("Number of Relevant Chunks (k)", min_value=1, max_value=5, value=3, step=1,
                      help="How many pieces of context are retrieved to answer the question.")
        
        # Add a clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # --- Main App Content ---
    st.title("Ask Your Medical Question")

    # Initialize session state for messages if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your medical questions today?"}]

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get user prompt
    prompt = st.chat_input("What is your question?")

    if prompt:
        # Add user message to session state and display it
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # --- Chatbot Logic ---
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        try: 
            # Show a spinner while processing
            with st.spinner("🧑‍⚕️ Thinking..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    # The error is already shown in get_vectorstore, so we just stop.
                    return

                qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': k}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]

                # --- Nicely Formatted Output ---
                with st.chat_message('assistant'):
                    st.markdown(result)
                    # Add the formatted sources in an expander
                    with st.expander("📚 View Sources"):
                        for doc in source_documents:
                            # Try to get the source from metadata, fallback to 'N/A'
                            source_name = doc.metadata.get('source', 'N/A').split(os.sep)[-1]
                            st.info(f"**Source:** `{source_name}`")
                            st.text_area("Content:", value=doc.page_content, height=150, disabled=True, key=f"doc_{source_documents.index(doc)}")
                
                # Add just the text result to session state for a cleaner history
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- Entry Point of the Script ---
if __name__ == "__main__":
    main()