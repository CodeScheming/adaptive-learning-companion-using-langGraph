import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app as langgraph_app
import uuid

# A dictionary to store the state of each user's session
# We use a unique session_id to track each conversation
session_states = {}

def get_session_id(session_id: str = None) -> str:
    """Returns a session ID."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    return session_id

def chat_interface(message, history, session_id: str = None):
    """
    This is the main function that interacts with the LangGraph agent.
    It receives a user message and the chat history, and returns the agent's response.
    """
    # Get or create a session ID
    session_id = get_session_id(session_id)
    
    # Define the LangGraph configuration for this session
    config = {"configurable": {"thread_id": session_id}}
    
    # Initialize state if it's a new session
    if session_id not in session_states:
        session_states[session_id] = {
            "messages": [],
            "current_topic": "Recursion in Python",
            "assessment": None,
            "attempts": 0
        }
        # This is the initial call to start the graph and get the first message
        initial_response = ""
        for event in langgraph_app.stream(session_states[session_id], config):
            for value in event.values():
                if 'messages' in value and value['messages']:
                    initial_response += value['messages'][-1].content + "\n"
        return initial_response.strip()

    # If it's an existing session, stream the user's message to the graph
    inputs = {"messages": [HumanMessage(content=message)]}
    response_generator = langgraph_app.stream(inputs, config)
    
    full_response = ""
    for event in response_generator:
        for value in event.values():
            if 'messages' in value and isinstance(value['messages'][-1], AIMessage):
                full_response += value['messages'][-1].content + " "

    # Check if the graph has finished
    current_state = langgraph_app.get_state(config)
    if not current_state.next:
        full_response += "\n\n**Congratulations! You've completed the topic. The session has ended.**"

    return full_response.strip()

# Create the Gradio Chat Interface
chatbot_ui = gr.ChatInterface(
    fn=chat_interface,
    title="🤖 Adaptive Learning Companion",
    description="Learn about 'Recursion in Python'. The tutor will guide you, ask you questions, and adapt to your answers.",
    examples=[["What is recursion?"], ["def factorial(n):\n  if n == 0:\n    return 1\n  else:\n    return n * factorial(n-1)"]],
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).queue() # Use queue for better user experience with streaming