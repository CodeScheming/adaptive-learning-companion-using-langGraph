import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field  # Corrected Pydantic import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- FIX STARTS HERE ---
# Load environment variables from .env at the start
load_dotenv()

# Explicitly pass the API key to the constructor
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize the Gemini model with the API key
# The corrected line
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
# --- FIX ENDS HERE ---

class Evaluation(BaseModel):
    """Represents the evaluation of a user's answer."""
    correct: bool = Field(description="Whether the user's answer was correct.")
    reasoning: str = Field(description="The reasoning behind the evaluation.")
    missing_concept: str = Field(description="The specific foundational concept the user might be missing, if any. e.g., 'base case', 'recursive step'.", default=None)

# Explainer Agent: Explains a concept to the user.
def explainer_agent(state):
    """
    Provides an explanation for the current topic.
    """
    system_prompt = (
        "You are an expert tutor. Your goal is to explain a complex topic in a simple, "
        "easy-to-understand way. The topic is: {topic}."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ]).format(topic=state['current_topic'])
    
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response.content)]}

# Quiz Master Agent: Creates a quiz question for the user.
def quiz_master_agent(state):
    """
    Generates a quiz question based on the current topic.
    """
    system_prompt = (
        "You are a quiz master. Create a simple, practical question to test the user's "
        "understanding of the following topic: {topic}. "
        "The user should be able to answer with a small code snippet or a short explanation."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ]).format(topic=state['current_topic'])
    
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response.content)]}

# Evaluator Agent: Assesses the user's answer.
def evaluator_agent(state):
    """
    Evaluates the user's answer and determines if it's correct.
    """
    structured_llm = llm.with_structured_output(Evaluation)
    
    user_message = state["messages"][-1].content
    quiz_question = state["messages"][-2].content

    system_prompt = (
        "You are an expert evaluator. Your task is to assess a user's answer to a quiz question. "
        "Analyze the user's answer for correctness and identify any missing foundational concepts. "
        "The topic is: {topic}.\n\n"
        "Quiz Question: {quiz_question}\n"
        "User's Answer: {user_answer}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ]).format(topic=state['current_topic'], quiz_question=quiz_question, user_answer=user_message)
    
    evaluation = structured_llm.invoke(prompt)
    
    new_attempts = state.get('attempts', 0) + 1
    
    if evaluation.correct:
        response = "That's correct! Great job."
        return {"assessment": "correct", "messages": [AIMessage(content=response)]}
    
    response = f"That's not quite right. {evaluation.reasoning}"
    
    # Decide the next step based on the evaluation
    if new_attempts > 1 and evaluation.missing_concept:
        assessment = "revisit_foundation"
    else:
        assessment = "provide_hint"
        
    return {
        "assessment": assessment, 
        "messages": [AIMessage(content=response)], 
        "attempts": new_attempts
    }

# Hint Provider Agent: Gives a hint if the user is stuck.
def hint_provider_agent(state):
    """
    Provides a hint to the user if their answer was incorrect.
    """
    user_message = state["messages"][-2].content
    quiz_question = state["messages"][-3].content
    
    system_prompt = (
        "You are a helpful teaching assistant. The user is stuck on a problem. "
        "Provide a targeted hint without giving away the answer. "
        "The topic is: {topic}.\n\n"
        "Quiz Question: {quiz_question}\n"
        "User's Incorrect Answer: {user_answer}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ]).format(topic=state['current_topic'], quiz_question=quiz_question, user_answer=user_message)
    
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response.content)]}

# Foundation Re-visitor Agent: Re-explains a foundational concept.
def foundation_revisitor_agent(state):
    """
    Briefly re-explains a foundational concept that the user is missing.
    """
    system_prompt = (
        "You are a tutor. The user is struggling with the current topic because they seem to be "
        "missing a foundational concept. Briefly re-explain this concept in a simple way. "
        "The core topic is: {topic}. The missing concept is likely related to the user's last incorrect answer."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ]).format(topic=state['current_topic'])
    
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=f"Let's take a step back and review a key idea.\n\n{response.content}\n\nNow, let's try that quiz question again.")]}