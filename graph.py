# ==============================================================================
# FILE OVERVIEW: graph.py (The Workflow Map / Traffic Controller)
# ==============================================================================
# If `state.py` is the shared clipboard and `nodes.py` contains the individual
# workers, then `graph.py` is the **Factory Floor Map** (powered by LangGraph)
# that defines the exact order in which the workers do their jobs.
#
# Key LangGraph Concepts Explained for a Layman:
# 1. `StateGraph`: A flowchart builder. It manages our shared clipboard (`MeetingState`)
#    and passes it from worker to worker.
# 2. `Nodes` (`add_node`): The stations/stops on our flowchart (the functions from `nodes.py`).
# 3. `Normal Edges` (`add_edge`): Fixed one-way arrows between stations.
#    For example: "After `preprocessor` finishes, ALWAYS go to `summarizer`."
# 4. `Conditional Edges` (`add_conditional_edges`): Decision forks in the road!
#    Instead of always going to the same next step, a small routing function
#    checks the clipboard and decides which way to go next:
#    - Fork 1 (`route_clarification`): Are any task owners or deadlines missing?
#      If YES -> detour to `human_clarification`. If NO -> go straight to `reviewer`.
#    - Fork 2 (`route_review_feedback`): Did the `reviewer` find mistakes?
#      If YES (and retries < 2) -> loop back to `action_extractor` to try again.
#      If NO (approved) -> proceed to `report_assembler`.
# 5. `END`: A special built-in marker that tells LangGraph the workflow is finished.
# ==============================================================================

from typing import Literal  # Restricts function return types to specific exact strings
from langgraph.graph import StateGraph, END  # Core LangGraph tools to build the workflow flowchart
from state import MeetingState  # Our shared clipboard blueprint
from nodes import (  # Import all 6 worker functions from nodes.py
    preprocessor_node,
    summarizer_node,
    action_extractor_node,
    human_clarification_node,
    reviewer_node,
    report_assembler_node
)


# ------------------------------------------------------------------------------
# Decision Fork 1: Should we ask a human for help, or go straight to review?
# ------------------------------------------------------------------------------
# After the `action_extractor` finishes finding tasks, LangGraph runs this function
# to decide which node to visit next.
def route_clarification(state: MeetingState) -> Literal["human_clarification", "reviewer"]:
    """Routes to human clarification if any action item has an unresolved owner or deadline."""
    # Loop through all extracted tasks on the clipboard
    for item in state.get("action_items", []):
        needs_owner = not item.owner or item.owner.strip().lower() in ["none", "null", "unassigned"]
        needs_deadline = not item.deadline or item.deadline.strip().lower() in ["none", "null", "unspecified"]
        
        # If even ONE task is missing an owner or deadline, detour to the human clarification step
        if needs_owner or needs_deadline:
            return "human_clarification"
            
    # Otherwise, if everything is already clear, skip the human prompt and go straight to the Reviewer AI
    return "reviewer"


# ------------------------------------------------------------------------------
# Decision Fork 2: Did the Reviewer approve the work, or do we need a redo?
# ------------------------------------------------------------------------------
# After the `reviewer` finishes fact-checking, LangGraph runs this function
# to decide whether to loop back for corrections or move forward to build the report.
def route_review_feedback(state: MeetingState) -> Literal["action_extractor", "report_assembler"]:
    """Allows at most 2 retry loops if reviewer finds discrepancies; otherwise proceeds to assembly."""
    feedback = state.get("review_feedback")
    retry_count = state.get("retry_count", 0)
    
    # Case 1: Approved by reviewer (no hallucinations or errors found) -> Move to final report assembly!
    if feedback and feedback.is_valid:
        return "report_assembler"
    
    # Case 2: Loop limit guard (capped at 2 retries)
    # Even if the Reviewer isn't happy, we stop looping after 2 attempts so the program
    # never gets stuck in an infinite loop burning API credits.
    if retry_count >= 2:
        print(f"\n[Notice] Reached maximum correction retries ({retry_count}). Proceeding to assembly.")
        return "report_assembler"
    
    # Case 3: Reviewer found an error and we still have retries left -> Send back to action_extractor!
    print(f"\n[Notice] Discrepancy detected by Reviewer (Retry {retry_count}/2). Routing back for corrections.")
    return "action_extractor"


# ------------------------------------------------------------------------------
# Graph Builder: Wiring the Flowchart Together
# ------------------------------------------------------------------------------
def build_meeting_graph():
    """Builds and compiles the StateGraph workflow."""
    # 1. Create a new blank flowchart that uses `MeetingState` as its shared clipboard
    workflow = StateGraph(MeetingState)
    
    # 2. Register Nodes (Add all 6 worker stations to the flowchart and give each a name)
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("action_extractor", action_extractor_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("report_assembler", report_assembler_node)
    
    # 3. Define Entry and Sequential Edges (Connect the fixed one-way arrows)
    # Start the workflow at 'preprocessor'
    workflow.set_entry_point("preprocessor")
    # From 'preprocessor', always go to 'summarizer'
    workflow.add_edge("preprocessor", "summarizer")
    # From 'summarizer', always go to 'action_extractor'
    workflow.add_edge("summarizer", "action_extractor")
    
    # 4. Conditional Branch: Check if any clarification is needed
    # After 'action_extractor', call `route_clarification` to pick the next stop:
    # either 'human_clarification' or 'reviewer'.
    workflow.add_conditional_edges(
        "action_extractor",
        route_clarification,
        {
            "human_clarification": "human_clarification",
            "reviewer": "reviewer"
        }
    )
    
    # After the human finishes answering questions in 'human_clarification', always go to 'reviewer'
    workflow.add_edge("human_clarification", "reviewer")
    
    # 5. Conditional Edge: Review verification loop (capped at 2)
    # After 'reviewer', call `route_review_feedback` to either loop back to 'action_extractor'
    # or move ahead to 'report_assembler'.
    workflow.add_conditional_edges(
        "reviewer",
        route_review_feedback,
        {
            "action_extractor": "action_extractor",
            "report_assembler": "report_assembler"
        }
    )
    
    # 6. Final Edge: After 'report_assembler' saves the files, mark the workflow as finished (END)
    workflow.add_edge("report_assembler", END)
    
    # 7. Compile the flowchart into an executable application and return it
    return workflow.compile()
