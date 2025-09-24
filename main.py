import os
from dotenv import load_dotenv
from app import chatbot_ui

# Load environment variables from .env file
load_dotenv()

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set. Please create a .env file.")
        return
        
    print("Launching Gradio UI...")
    # The launch() method creates a web server and provides a public URL if share=True
    chatbot_ui.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()