import streamlit as st
from jamaibase import JamAI, protocol as p
import time
import os

# Initialize session state variables
def initialize_session_state():
    if "unique_time" not in st.session_state:
        st.session_state.unique_time = time.time()
    if "knowledge_base_exist" not in st.session_state:
        st.session_state.knowledge_base_exist = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "project_id" not in st.session_state:
        st.session_state.project_id = None
    if "model" not in st.session_state:
        st.session_state.model = "ellm/Qwen/Qwen2-7B-Instruct"
    if "k" not in st.session_state:
        st.session_state.k = 2
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.01
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.01
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 496

# Function to check if any input is None
def check_all_fields_filled(file_upload, api_key, proj_id):
    return file_upload is not None and api_key and proj_id

def clear_credentials():
    st.session_state.api_key = ""
    st.session_state.project_id = ""
    st.session_state.unique_time = time.time() # reset the unique time
    
# Function to create knowledge base
def create_knowledge_base(jamai, file_upload):
    try:
        with st.spinner("Creating Knowledge Base..."):
            knowledge_simple = f"knowledge-simple-{st.session_state.unique_time}"
            knowledge_table = jamai.create_knowledge_table(
                p.KnowledgeTableSchemaCreate(
                    id=knowledge_simple,
                    cols=[],
                    embedding_model="ellm/BAAI/bge-m3",
                )
            )
        st.success("Successfully created Knowledge Base")

        # Save PDF to local directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.read())

        # Upload file to knowledge base
        with st.spinner("Uploading PDF to Knowledge Base..."):
            response = jamai.upload_file(
                p.FileUploadRequest(
                    file_path=file_path,
                    table_id=knowledge_simple,
                )
            )
            assert response.ok
        st.success("Successfully uploaded PDF to Knowledge Base!")
        st.session_state.knowledge_base_exist = True
        
        # remove after upload successfully
        os.remove(file_path)

        return knowledge_simple
    except Exception as e:
        clear_credentials()
        # st.error(f"An error occurred: {str(e)}")
        st.warning("An error occurred. Please check your credentials and try again.")
        return None
    
# Function to create chat table
def create_chat_table(jamai, knowledge_simple):
    try:
        with st.spinner("Creating Chat Table..."):
            table = jamai.create_chat_table(
                p.ChatTableSchemaCreate(
                    id=f"chat-rag-{st.session_state.unique_time}",
                    cols=[
                        p.ColumnSchemaCreate(id="User", dtype=p.DtypeCreateEnum.str_),
                        p.ColumnSchemaCreate(
                            id="AI",
                            dtype=p.DtypeCreateEnum.str_,
                            gen_config=p.ChatRequest(
                                model=st.session_state.model,
                                messages=[p.ChatEntry.system("You are a concise assistant.")],
                                rag_params=p.RAGParams(
                                    table_id=knowledge_simple,
                                    k=st.session_state.k,
                                ),
                                temperature=st.session_state.temperature,
                                top_p=st.session_state.top_p,
                                max_tokens=st.session_state.max_tokens,
                            ).model_dump(),
                        ),
                    ],
                )
            )
        st.success("Successfully created Chat Table")
    except Exception as e:
        clear_credentials()
        # st.error(f"An error occurred while creating the chat table: {str(e)}")
        st.warning("An error occurred. Please check your credentials and try again.")

# Function to ask a question with improved streaming output
def ask_question(question):
    jamai = JamAI(api_key=st.session_state.api_key, project_id=st.session_state.project_id)
    completion = jamai.add_table_rows(
        "chat",
        p.RowAddRequest(
            table_id=f"chat-rag-{st.session_state.unique_time}",
            data=[dict(User=question)],
            stream=True,
        ),
    )
    
    full_response = ""

    for chunk in completion:
        if chunk.output_column_name != "AI":
            continue
        if isinstance(chunk, p.GenTableStreamReferences):
            pass
        else:
            full_response += chunk.text
            yield full_response

# Main app function
def main():
    st.title("Chat With PDF")

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.logo("resource/Jamai-Long-Black-Main.icuEAbYB.svg")
        login_button = """
            <a href="https://cloud.jamaibase.com/" style="
            display: inline-block;
            padding: 10px 30px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            transition: 0.25s;
            color: #ffffff;
            background-color: #007bff;
            border: 1px solid transparent;
            border-radius: 0.25rem;
            ">
            Login to your Jamai Base
            </a>
        """

        st.markdown(login_button, unsafe_allow_html=True)
        
        st.header("Configuration")
        file_upload = st.file_uploader("Upload your PDF", type=["pdf"], disabled=st.session_state.knowledge_base_exist)
        api_key = st.text_input('JAMAI API KEY', value=st.session_state.get('api_key', ''), type='password', disabled=st.session_state.knowledge_base_exist)
        project_id = st.text_input('Project ID', value=st.session_state.get('project_id', ''), disabled=st.session_state.knowledge_base_exist)
        
        with st.expander("Advanced Settings"):
            model_options = [
                            'ellm/Qwen/Qwen2-7B-Instruct', 'ellm/Qwen/Qwen2-72B-Instruct', 
                            'ellm/meta-llama/Llama-3-8B-Instruct', 'ellm/meta-llama/Llama-3-70B-Instruct', 
                            'ellm/meta-llama/Llama-3.1-8B-Instruct', 'ellm/meta-llama/Llama-3.1-70B-Instruct',
                            'ellm/microsoft/Phi-3-mini-128k-Instruct-Int4'
                            'ellm/microsoft/Phi-3-medium-128k-Instruct-Int4',
                            ]
            st.session_state.model = st.selectbox("Model", options=model_options, index=0, disabled=st.session_state.knowledge_base_exist)
            st.session_state.k = st.slider("k", value=2, min_value=1, max_value=30, step=1, disabled=st.session_state.knowledge_base_exist)
            st.session_state.max_tokens = st.slider("max tokens", value=496, min_value=96, max_value=960, step=8, disabled=st.session_state.knowledge_base_exist)
            temperature_options = [str(i/10) for i in range(1,11)]
            st.session_state.temperature = st.selectbox("temperature", options=temperature_options, format_func=lambda x: float(x), disabled=st.session_state.knowledge_base_exist)
            top_p_options = [str(i/10) for i in range(1,11)]
            st.session_state.top_p = st.selectbox("top p", options=top_p_options, format_func=lambda x: float(x), disabled=st.session_state.knowledge_base_exist)

        if st.button("Create Knowledge Base", disabled=st.session_state.knowledge_base_exist):
            if not check_all_fields_filled(file_upload, api_key, project_id):
                st.warning("Please provide all required information: PDF file, JAMAI API KEY, and Project ID.")
            else:
                st.session_state.api_key = api_key
                st.session_state.project_id = project_id
                jamai = JamAI(api_key=api_key, project_id=project_id)
                knowledge_simple = create_knowledge_base(jamai, file_upload)
                if knowledge_simple:
                    create_chat_table(jamai, knowledge_simple)

    # Main chat interface
    st.header("Jamai Base")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your PDF"):
        if st.session_state.knowledge_base_exist:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in ask_question(question):
                    message_placeholder.markdown(response + "▌")
                    full_response = response
                message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please upload a PDF and create a Knowledge Base first.")

if __name__ == "__main__":
    main()