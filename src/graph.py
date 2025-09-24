from langgraph.graph import StateGraph, END
from .state import TutorState
from .agents import (
    explainer_agent, 
    quiz_master_agent, 
    evaluator_agent, 
    hint_provider_agent, 
    foundation_revisitor_agent
)

# Define the graph
workflow = StateGraph(TutorState)

# Add nodes to the graph
workflow.add_node("explainer", explainer_agent)
workflow.add_node("quiz_master", quiz_master_agent)
workflow.add_node("evaluator", evaluator_agent)
workflow.add_node("hint_provider", hint_provider_agent)
workflow.add_node("foundation_revisitor", foundation_revisitor_agent)

# Set the entry point of the graph
workflow.set_entry_point("explainer")

# Define the edges

# From explainer, go to quiz master
workflow.add_edge("explainer", "quiz_master")

# Conditional routing after evaluation
def route_after_evaluation(state):
    if state["assessment"] == "correct":
        return END
    elif state["assessment"] == "revisit_foundation":
        return "foundation_revisitor"
    else:
        return "hint_provider"

workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "hint_provider": "hint_provider",
        "foundation_revisitor": "foundation_revisitor",
        END: END,
    },
)

# After getting a hint or revisiting a foundation, loop back to the evaluator
workflow.add_edge("hint_provider", "evaluator")
workflow.add_edge("foundation_revisitor", "evaluator")

# Compile the graph into a runnable object
app = workflow.compile()

# This allows us to see the graph structure
try:
    # Generate the image
    img = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
    print("Graph visualization saved as graph.png")
except Exception as e:
    print(f"Could not create graph visualization: {e}")