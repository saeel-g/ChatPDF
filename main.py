import streamlit as st
import pickle
import pdfplumber as pd
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

#creating Sidebar
with st.sidebar:
    st.title("Chat with PDF 🗣️")
    st.markdown('''
    ## About
    This is a LLM power bot build to answer your question from given PDF
    \n Made by [saeel-g](https://github.com/saeel-g)
    \n Find the code on [GitHub](https://github.com/saeel-g/ChatPDF-Bot)
''')
    
api_key = 'auth_key'

def main():
    st.header('Chat with PDF 🗣️')
    pdf_file = st.file_uploader("Upload your PDF file", type='pdf')
   
   # Reading the pdf and extracting the text
    if pdf_file is not None:
        read_pdf= pd.open(pdf_file)
        text=''
        for page in read_pdf.pages:
            text += page.extract_text()

        #Splitting the text into Chunks
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunk=text_splitter.split_text(text=text)
        # st.write(chunk)
       
        # Embeddings
        store_name=pdf_file.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f'{store_name}.pkl',"rb") as f:
                VectorStore=pickle.load(f)
            # st.write('Embeddings loaded')
        
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunk, embedding=embeddings)
            with open(f"{store_name}.pkl", 'wb') as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings Created")

        #Accepting user's question
       
        query=st.text_input("please ask a question regarding given PDF:")
        if query:
            docs=VectorStore.similarity_search(query=query, k=3) 
            llm=OpenAI(api_key=api_key,model_name='gpt-3.5-turbo') 
            chain=load_qa_chain(llm=llm, chain_type='stuff')
            response=chain.run(input_documents=docs, question=query)
            st.write(response)



if __name__=='__main__':
    main()

