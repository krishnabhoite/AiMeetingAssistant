# ==============================================================================
# FILE OVERVIEW: graph.py (The Workflow Map / Traffic Controller)
# ==============================================================================
# If `state.py` is the shared clipboard and `nodes.py` contains the individual
# workers, then `graph.py` is the **Factory Floor Map** (powered by LangGraph)
# that defines the exact order in which the workers do their jobs.
#
# How the Workflow Runs Step-by-Step:
#   1. `preprocessor`: Numbers each line of the transcript (L01, L02...).
#   2. `summarizer`: Summarizes the discussion and extracts key decisions.
#   3. `action_extractor`: Extracts action items, owners, and deadlines.
#   4. `reviewer`: Checks the summary, decisions, and action items against the
#      transcript. Flags any unsupported statements and requests corrections.
#   5. Decision Fork (`route_after_review`):
#      - If the Reviewer found unsupported statements AND we haven't done a
#        correction pass yet (retry_count == 1), route back ONCE to `summarizer`
#        -> `action_extractor` to fix the mistakes before asking the human.
#      - Once verified (or after the single correction pass), check if any
#        action item is missing an owner or deadline:
#        * If YES -> go to `human_clarification` to ask the user (only ONCE!).
#        * If NO  -> go straight to `report_assembler`.
#   6. `report_assembler`: Formats and exports the final reviewed report and
#      agent activity log, then finishes at `END`.
# ==============================================================================

from typing import Literal  # Restricts function return types to specific exact strings
from langgraph.graph import StateGraph, END  # Core LangGraph tools to build the workflow flowchart
from state import MeetingState  # Our shared clipboard blueprint
from nodes import (  # Import all 6 worker functions from nodes.py
    preprocessor_node,
    summarizer_node,
    action_extractor_node,
    reviewer_node,
    human_clarification_node,
    report_assembler_node
)


# ------------------------------------------------------------------------------
# Decision Fork: What happens after the Reviewer finishes checking the AI's work?
# ------------------------------------------------------------------------------
def route_after_review(
    state: MeetingState
) -> Literal["summarizer", "human_clarification", "report_assembler"]:
    """
    Decides the next step after the Reviewer node:
    1. If Reviewer found unsupported statements on the first pass (retry_count == 1),
       route back ONCE to 'summarizer' -> 'action_extractor' for correction.
    2. Otherwise, if any action item is missing an owner or deadline, route to
       'human_clarification' to ask the user.
    3. Otherwise, proceed directly to 'report_assembler'.
    """
    feedback = state.get("review_feedback")
    retry_count = state.get("retry_count", 0)

    # Case 1: The Reviewer flagged unsupported statements and we haven't done our
    # single correction pass yet (retry_count == 1). Route back once to fix them!
    if feedback and not feedback.is_valid and retry_count <= 1:
        print(
            f"\n[Review Notice] Unsupported statement flagged by Reviewer. "
            f"Running 1 correction pass before final report..."
        )
        return "summarizer"

    # Case 2: Review is complete! Now check if any task has a missing owner or deadline
    # so we can ask the human user for clarification (at most once).
    for item in state.get("action_items", []):
        needs_owner = not item.owner or item.owner.strip().lower() in ["none", "null", "unassigned", "unresolved"]
        needs_deadline = not item.deadline or item.deadline.strip().lower() in ["none", "null", "unspecified", "unresolved"]

        # If even ONE task is missing an owner or deadline, detour to Human Clarification
        if needs_owner or needs_deadline:
            return "human_clarification"

    # Case 3: Everything is verified and all tasks have owners and deadlines -> Assemble Report!
    return "report_assembler"


# ------------------------------------------------------------------------------
# Graph Builder: Wiring the Flowchart Together
# ------------------------------------------------------------------------------
def build_meeting_graph():
    """Builds and compiles the LangGraph multi-agent workflow."""
    # Step 1: Create a new blank flowchart that uses `MeetingState` as its shared clipboard
    workflow = StateGraph(MeetingState)
    
    # Step 2: Register all 6 worker stations (nodes) on the flowchart
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("action_extractor", action_extractor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("report_assembler", report_assembler_node)
    
    # Step 3: Connect the starting station and sequential one-way arrows
    # Start the workflow at 'preprocessor'
    workflow.set_entry_point("preprocessor")
    # After 'preprocessor' numbers the lines, always go to 'summarizer'
    workflow.add_edge("preprocessor", "summarizer")
    # After 'summarizer' writes the summary & decisions, always go to 'action_extractor'
    workflow.add_edge("summarizer", "action_extractor")
    # After 'action_extractor' finds tasks, always go to 'reviewer' to fact-check against transcript
    workflow.add_edge("action_extractor", "reviewer")
    
    # Step 4: Conditional Branch after 'reviewer'
    # Calls `route_after_review` to either:
    # - Run a single correction pass ('summarizer'),
    # - Ask the user to clarify missing task owners/deadlines ('human_clarification'), or
    # - Build the final report ('report_assembler').
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "summarizer": "summarizer",
            "human_clarification": "human_clarification",
            "report_assembler": "report_assembler"
        }
    )
    
    # Step 5: After the human answers clarification questions, always go to 'report_assembler'
    workflow.add_edge("human_clarification", "report_assembler")
    
    # Step 6: After 'report_assembler' exports the files, mark the workflow as finished (END)
    workflow.add_edge("report_assembler", END)
    
    # Step 7: Compile the flowchart into a runnable application and return it
    return workflow.compile()
