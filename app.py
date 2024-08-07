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

# Function to check if any input is None
def check_none(*args):
    return all(arg is not None for arg in args)

# Function to create knowledge base
def create_knowledge_base(jamai, file_upload):
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

# Function to create chat table
def create_chat_table(jamai, knowledge_simple):
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
                            model="ellm/Qwen/Qwen2-7B-Instruct",
                            messages=[p.ChatEntry.system("You are a concise assistant.")],
                            rag_params=p.RAGParams(
                                table_id=knowledge_simple,
                                k=2,
                            ),
                            temperature=0.001,
                            top_p=0.001,
                            max_tokens=500,
                        ).model_dump(),
                    ),
                ],
            )
        )
    st.success("Successfully created Chat Table")

# Function to ask a question with improved streaming output
def ask_question(jamai, question):
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
    st.title("Chat With Your PDF")

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        file_upload = st.file_uploader("Upload your PDF", type=["pdf"])
        JAMAI_API_KEY = st.text_input('JAMAI API KEY', type='password')
        PROJ_ID = st.text_input('Project ID')

        if st.button("Create Knowledge Base"):
            if not check_none(file_upload, JAMAI_API_KEY, PROJ_ID):
                st.warning("Please provide all required information")
            else:
                jamai = JamAI(api_key=JAMAI_API_KEY, project_id=PROJ_ID)
                knowledge_simple = create_knowledge_base(jamai, file_upload)
                create_chat_table(jamai, knowledge_simple)

    # Main chat interface
    st.header("Chat")

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
                jamai = JamAI(api_key=JAMAI_API_KEY, project_id=PROJ_ID)
                full_response = ""
                for response in ask_question(jamai, question):
                    message_placeholder.markdown(response + "▌")
                    full_response = response
                message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please upload a PDF and create a Knowledge Base first.")

if __name__ == "__main__":
    main()