import gradio as gr
import os
from pathlib import Path
from reliable_rag import DocumentProcessor, GeminiChat, ProcessingConfig, ChatConfig

ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md']

# Initialize processors
processing_config = ProcessingConfig(collection_name='lokesh')
chat_config = ChatConfig(
    system_prompt="""You are a helpful RAG assistant. 
    When user asks you a question, answer only based on given context. 
    If you can't find an answer based on context reply "I don't know".
    """
)

doc_processor = DocumentProcessor(processing_config)
chat_processor = GeminiChat(chat_config)

# Set up collection
doc_processor.setup_collection()

def process_uploaded_files(files):
    """Process uploaded files and add them to the vector store"""
    if not files:
        return "No files uploaded."
    
    if len(files) > 5:
        return "Error: Maximum 5 files can be processed at once."
    
    status_messages = []
    for file in files:
        file_path = Path(file.name)
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            status_messages.append(f"Skipped {file_path.name}: Unsupported file type")
            continue
            
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            doc_processor.process_document(str(file_path))
            status_messages.append(f"Successfully processed: {file_path.name}")
        except Exception as e:
            status_messages.append(f"Error processing {file_path.name}: {str(e)}")
    
    return "\n".join(status_messages)

def process_message(message: str, history: list) -> tuple[list, str]:
    """Process incoming messages using RAG"""
    try:
        search_results = doc_processor.hybrid_search(
            query_text=message,
            limit=5
        )
        
        if not search_results:
            return (history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "No relevant information found in the documents."}
            ], "No relevant context found")
        
        # Prepare context from search results
        context = "\n".join(search_results)
        
        response = chat_processor.send_message(message, context)
        
        return (history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ], context)
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return (history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ], "Error occurred while retrieving context")

def clear_chat_history():
    chat_processor.clear_chat()
    return []

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Hybrid RAG")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload PDF Documents")
            file_output = gr.Textbox(label="Upload Status", lines=2)
            files = gr.File(
                label="Upload Documents (max 5 files)",
                file_count="multiple",
                type="filepath"
            )
            upload_btn = gr.Button("Process Uploaded Files")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Message", placeholder="Type your question here...")
            clear_btn = gr.Button("Clear Chat History")
        with gr.Column(scale=1):
            context_box = gr.Textbox(
                label="Retrieved Context",
                placeholder="Context from documents will appear here...",
                lines=20,
                interactive=False
            )
    
    upload_btn.click(
        fn=process_uploaded_files,
        inputs=[files],
        outputs=[file_output]
    )
    
    msg.submit(
        fn=process_message, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, context_box]
    ).then(
        fn=lambda: "", 
        outputs=msg
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, context_box]
    )

if __name__ == "__main__":
    demo.launch(share=True)
