# ==============================================================================
# FILE OVERVIEW: AIAgentMeetingAssistant.py (Main Entry Point / Program Runner)
# ==============================================================================
# This is the main "Start Button" file that you run to execute the entire
# AI Meeting Assistant application.
#
# What this file does step-by-step:
#   1. Defines a sample meeting transcript (`SAMPLE_TRANSCRIPT`).
#   2. Builds the LangGraph workflow (`build_meeting_graph()` from `graph.py`).
#   3. Prepares a fresh, blank shared clipboard (`initial_state`) containing the
#      sample transcript.
#   4. Runs the workflow in real-time (`app.stream(...)`), printing each worker
#      station (node) as it runs.
#   5. Prints out the exact path the workflow took, the final Markdown report,
#      and visual diagrams (ASCII and Mermaid) of the workflow graph.
# ==============================================================================

import os  # Standard Python module for operating system interactions
from dotenv import load_dotenv  # Loads secret API keys from the .env file
from graph import build_meeting_graph  # Imports our flowchart builder function from graph.py
import argparse  # Standard Python module for parsing command-line arguments

# Load environment variables from .env before starting
load_dotenv()

# ------------------------------------------------------------------------------
# Sample Input Data (A short conversation from a team meeting)
# ------------------------------------------------------------------------------
# Notice that lines 11 and 12 intentionally have missing details ("someone to
# review security" has no owner/deadline; "migrate the auth service" has no owner)
# so that you can see the Human-in-the-Loop clarification feature in action!
# SAMPLE_TRANSCRIPT = """
# Ravi: We agreed in today's architecture sync to proceed with PostgreSQL for the database layer.
# Priya: Sounds good. I will update the design document by Monday.
# Ravi: I will complete the API testing by Friday.
# Ravi: We also need someone to review security.
# Priya: Yes, and we definitely need to migrate the auth service before the release.
# """

def load_transcript(file_path: str) -> str:
    """Reads a meeting transcript text file and returns its trimmed content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    if not content:
        raise ValueError(f"Transcript file '{file_path}' is empty.")
        
    return content


def parse_args():
    """Parses the transcript file path argument from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent AI Meeting Assistant with a transcript file."
    )
    parser.add_argument(
        "transcript_path",
        nargs="?",
        default="test_case_transcripts/tc_01_happy_path.txt",
        help="Path to the transcript text file (e.g., test_case_transcripts/tc_01_happy_path.txt)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    transcript_text = load_transcript(args.transcript_path)

    print("\n" + "=" * 65)
    print("Multi-Agent AI Meeting Assistant Starting...")
    print(f"Loaded Transcript: {args.transcript_path}")
    print("=" * 65)

    # Step 1: Build and compile the LangGraph workflow (wires together all nodes and edges)
    app = build_meeting_graph()

    # Step 2: Set up the initial Shared Clipboard (MeetingState)
    # We put our `SAMPLE_TRANSCRIPT` into `raw_transcript` and leave all other fields
    # blank/empty so the AI workers can fill them in as the workflow runs.
    initial_state = {
        "raw_transcript": transcript_text,
        "indexed_transcript": "",
        "summary": "",
        "decisions": [],
        "action_items": [],
        "review_feedback": None,
        "retry_count": 0,
        "final_report": "",
        "activity_log": []
    }

    # Keep a list of every node that runs, in chronological order, to display later
    executed_nodes_sequence = []
    final_state = initial_state

    # Step 3: Stream real-time node executions
    # LAYMAN NOTE on `app.stream(initial_state, stream_mode="updates")`:
    # Instead of running silently and only giving us the answer at the very end,
    # `.stream(..., stream_mode="updates")` lets us watch the assembly line live!
    # Every time a single worker (node) finishes its turn, it yields an `event`
    # telling us which node just finished (`node_name`) and what changes it wrote
    # to the clipboard (`node_output`).
    print("\n--- Real-Time Execution Trace (Showing Which Agent Performed Each Task) ---")
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            executed_nodes_sequence.append(node_name)
            
            # Display which agent just ran and what task it completed
            if isinstance(node_output, dict) and node_output.get("activity_log"):
                for log_entry in node_output["activity_log"]:
                    print(f"[*] Node '{node_name}' | Agent: [{log_entry.agent_name}] -> {log_entry.action} ({log_entry.details})")
            else:
                print(f"[*] Completed Node: -> {node_name}")
            
            # Merge the latest updates from this node into our local copy of `final_state`
            if isinstance(node_output, dict):
                final_state = {**final_state, **node_output}

    print("-" * 65)

    # Step 4: Display chronological path taken during runtime
    # Shows the exact sequence of stations visited (e.g., preprocessor -> summarizer -> ...)
    print("\n--- Actual Execution Path ---")
    print(" -> ".join(executed_nodes_sequence) + " -> END")
    print("-" * 65 + "\n")

    # Step 5: Print Final Report to the terminal screen
    print("#" * 65)
    print("FINAL DELIVERABLE REPORT")
    print("#" * 65 + "\n")
    print(final_state.get("final_report", "No report generated."))
    print("\n" + "#" * 65)

    # Step 6: Display Graph Topology at the end
    # LangGraph can automatically draw a visual map of the workflow in two formats:
    # 1. ASCII Art (text drawing right in the terminal)
    # 2. Mermaid syntax (can be pasted into Markdown or mermaid.live to view a flowchart)
    print("\n" + "=" * 65)
    print("GRAPH TOPOLOGY (ASCII)")
    print("=" * 65)
    try:
        print(app.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not render ASCII diagram (missing grandalf): {e}")

    print("\n" + "=" * 65)
    print("GRAPH TOPOLOGY (MERMAID)")
    print("=" * 65)
    print(app.get_graph().draw_mermaid())
    print("=" * 65 + "\n")


# Standard Python entry point: Only run `main()` if this file is executed directly
if __name__ == "__main__":
    main()
