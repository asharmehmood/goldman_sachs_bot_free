import langchain
from langchain.embeddings import HuggingFaceEmbeddings
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamlitCallbackHandler

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-v0.1"
# print("Loading model...")
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "min_length":1000}
)
print("Model loaded...")
# loader = DirectoryLoader('./course_text')
# docs = loader.load()
embeddings = HuggingFaceEmbeddings()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap  = 100,
#     length_function = len,
# )

# text_chunks = text_splitter.split_documents(docs)
# print(len(text_chunks))
# persist_directory = './faiss/'
# db = FAISS.from_documents(text_chunks, embeddings)
# print('embeddings generated')
# db.save_local(persist_directory)
# print("db saved")



def main():
    data = FAISS.load_local("./faiss/",embeddings)

    # Prompt
    template = """context: {context}
    Question: {question}"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=data.as_retriever(),
                                            memory=memory,
                                            verbose=True,
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    if 'chain' not in st.session_state:
        st.session_state['chain'] = qa_chain

    if user_msg := st.chat_input():
        st.chat_message("user").write(user_msg)
        with st.chat_message("assistant"):
            st.write("I'm thinking...")
            # st.spinner('Generating response...')
            st_callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.chain.run(query=user_msg) #,callbacks=[st_callback]
            # print(response)
            st.write(response)
            
if __name__ == "__main__":
    main()