import streamlit as st
import importlib
from dotenv import load_dotenv
from back import ChatBotBackend
import os


st.set_page_config(
    page_title='gpt but you are sam altman',
    page_icon='🌻',
    layout="wide"
)


load_dotenv()
api_key = os.getenv('HF_API_KEY')

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatBotBackend(api_key)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# main
st.title('Sam Saltman GPT')

#sidebar with upload

with st.sidebar:
    st.header('Upload Documents')
    uploaded_file = st.file_uploader('Add documents to chat with', type=['pdf'])

    if uploaded_file:
        file_name = uploaded_file.name

        if "processed_file" not in st.session_state or st.session_state.processed_file != file_name:
            with st.spinner("Loooooooooooooooooading"):
                
                document_processor = importlib.import_module('doc')
                st.session_state.documents = document_processor.extract_text_from_file(uploaded_file)

                if st.session_state.documents:
                    document_text = st.session_state.documents[0].page_content

                # chunk the doc 
                st.session_state.chunks, st.session_state.metadatas = document_processor.chunk_text(st.session_state.documents)
                st.write("Chunks received")

                #add documents to chatbot
                st.session_state.chatbot.create_vector_store(st.session_state.chunks, st.session_state.metadatas)
                
                # mark files as processed
                st.session_state.processed_file = file_name
                st.success(f'Document processed: {file_name}')

    # settings
    st.header('Settings')
    use_context = st.checkbox('Use document context', value=True)

    #ahbout section
    st.header('About')
    st.write('''
            This Chatbot uses:
        - T5/FLAN-Base for responses
        - Sentence Transformers for embeddings
        - FAISS for vector search
        ''')     

#display chat 
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message['content'])

#chat input
user_input = st.chat_input('Ask me nothing, I like to sit idle')

if user_input:
    # add user message to chat history
    st.session_state.chat_history.append({
        "role": 'user', 
        'content': user_input
    })

    #display user message
    with st.chat_message('user'):
        st.write(user_input)
    
    #get response from the bot
    with st.chat_message('assistant'):
        with st.spinner('Dreaming'):
            response = st.session_state.chatbot.get_answer(
                user_input, 
                st.session_state.chunks, 
                use_context=use_context
            )

        st.write(response)

    # add assistant response to history
    st.session_state.chat_history.append({
        "role": 'assistant', 
        'content': response
    })
        