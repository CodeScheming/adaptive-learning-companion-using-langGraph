from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The state for our graph, which will be passed between nodes
class TutorState(TypedDict):
    # A list of messages that is appended to, rather than overwritten
    messages: Annotated[List[BaseMessage], add_messages]
    # The current topic the user is learning
    current_topic: str
    # A flag to determine if the user's answer was correct
    assessment: str
    # The number of attempts the user has made on the current quiz
    attempts: int