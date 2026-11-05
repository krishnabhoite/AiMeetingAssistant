# ==============================================================================
# FILE OVERVIEW: nodes.py (The AI Workers / Stations on the Assembly Line)
# ==============================================================================
# In LangGraph, a "Node" is simply a Python function that performs one specific
# job. Imagine an office assembly line where each desk has a specialist:
#   1. Preprocessor Node: Adds line numbers (L01, L02...) to the raw transcript.
#   2. Summarizer Node: An AI that writes an executive summary & lists decisions.
#   3. Action Extractor Node: An AI that finds to-do tasks, owners, and deadlines.
#   4. Reviewer Node: A "Fact-Checker" AI that checks summary, decisions, and tasks
#      against the transcript, flags unsupported statements, and requests corrections.
#   5. Human Clarification Node: Pauses and asks a real human (via keyboard input)
#      to fill in any missing task owners or deadlines (or marks them 'Unresolved').
#   6. Report Assembler Node: Formats everything into neat tables (including the
#      brief agent activity log) and saves the final report and audit log files.
#
# How a Node Works:
# - INPUT: Every node receives the current `state` (the shared clipboard from state.py).
# - OUTPUT: Every node returns a Python dictionary `{...}` containing ONLY the fields
#   it wants to update on the shared clipboard.
#
# Key LangChain Concepts Explained for a Layman:
# - `ChatOpenAI`: The connection to the AI brain (Large Language Model / LLM).
# - `ChatPromptTemplate`: A reusable instruction sheet with fill-in-the-blank
#   placeholders (like `{indexed_transcript}`) that we send to the AI.
# - `llm.with_structured_output(...)`: Forces the AI to reply using our strict
#   Pydantic forms (from state.py) instead of rambling in plain text.
# - `prompt | structured_llm`: The pipe symbol (`|`) in LangChain connects steps
#   together like Lego blocks: "Take the prompt, fill in the blanks, and feed it
#   directly into the AI model."
# - `chain.invoke({...})`: Actually presses the "Run" button on that connected chain.
# ==============================================================================

import os  # Used to read secret API keys stored in the computer's environment
from dotenv import load_dotenv  # Loads secret keys from a hidden '.env' file
from langchain_openai import ChatOpenAI  # LangChain wrapper to talk to OpenAI-compatible AI models
from langchain_core.prompts import ChatPromptTemplate  # Tool to build structured instructions (prompts) for the AI
from tabulate import tabulate  # Formats Python lists into clean, readable Markdown tables
from datetime import datetime  # Gets the current date and time for timestamps
from state import (  # Importing our shared state and structured data blueprints
    MeetingState,
    SummarizerOutput,
    ActionExtractorOutput,
    ReviewFeedback,
    AgentLog
)

# Load environment variables (like OPENROUTER_API_KEY and OPEN_API_BASE) from the .env file
load_dotenv()

# ------------------------------------------------------------------------------
# AI Model (LLM) Setup
# ------------------------------------------------------------------------------
# Here we configure the AI "brain" that our nodes will use.
# - model="openai/gpt-4o-mini": The specific AI model we are calling.
# - temperature=0.0: Controls creativity. Setting it to 0.0 makes the AI strictly
#   factual, consistent, and deterministic (no wild guessing).
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPEN_API_BASE"),
    temperature=0.0
)


# ==============================================================================
# NODE 1: Preprocessor (Line Numbering Worker)
# ==============================================================================
# Why do we need this step?
# The requirements ask us to reference transcript line numbers for decisions and
# action items. By tagging every line with "L01:", "L02:", etc. up front, all
# downstream AI agents can cite exact line numbers accurately.
# Notice this node doesn't even need AI—it uses simple Python text formatting!
def preprocessor_node(state: MeetingState) -> dict:
    """Indexes each line of the raw transcript with line numbers (L01, L02, etc.)."""
    # Step 1: Split the raw transcript by line breaks, trim extra spaces, and skip empty lines
    lines = [line.strip() for line in state["raw_transcript"].strip().split("\n") if line.strip()]
    
    # Step 2: Attach a two-digit line label (L01, L02, L03...) to the front of each line
    indexed_lines = [f"L{idx+1:02d}: {line}" for idx, line in enumerate(lines)]
    
    # Step 3: Join all the numbered lines back into one single block of text
    indexed_transcript = "\n".join(indexed_lines)
    
    # Step 4: Return updates to the shared clipboard (MeetingState):
    # - Save the newly numbered transcript
    # - Reset retry_count to 0
    # - Add an entry to activity_log recording that Preprocessor finished its job
    return {
        "indexed_transcript": indexed_transcript,
        "retry_count": 0,
        "activity_log": [
            AgentLog(
                agent_name="Preprocessor",
                action="Index Transcript",
                details=f"Numbered {len(lines)} lines of text (L01 to L{len(lines):02d})."
            )
        ]
    }


# ==============================================================================
# NODE 2: Summarizer Agent (Summary & Decision Extractor)
# ==============================================================================
# This worker reads the numbered transcript and extracts:
#   1. A high-level executive summary of the meeting's main topics and discussion.
#   2. A list of concrete decisions made, along with the exact line number (e.g., L03).
# If the Reviewer AI previously flagged an unsupported statement, this node also
# reads the Reviewer's feedback notes and corrects the summary/decisions.
def summarizer_node(state: MeetingState) -> dict:
    """Extracts the high-level meeting summary and decisions with line citations."""
    # Step 1: Tell the AI that its answer MUST match the 'SummarizerOutput' form from state.py
    structured_llm = llm.with_structured_output(SummarizerOutput)
    
    # Step 2: Check if we are re-running this step because the Reviewer AI requested a correction.
    # If so, attach the Reviewer's feedback notes so the AI fixes the issue.
    critique_prompt_addition = ""
    if state.get("review_feedback") and not state["review_feedback"].is_valid:
        critique_prompt_addition = (
            f"\nIMPORTANT - PREVIOUS REVIEWER CORRECTION REQUEST TO FIX:\n"
            f"{state['review_feedback'].feedback_notes}\n"
        )

    # Step 3: Create the instructions (prompt) for the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Technical Meeting Summarizer Agent. Given the indexed transcript below, extract:\n"
         "1. summary: A concise overview of the meeting's main discussion topics and context.\n"
         "2. decisions: A list of concrete group decisions or architectural/process agreements made "
         "(do not list individual to-do tasks as decisions unless they represent a team decision), "
         "with the exact line label (e.g., L03) where the decision appears.\n"
         "Do NOT invent or extrapolate anything beyond what is explicitly stated in the transcript.\n"
         f"{critique_prompt_addition}"),
        ("user", "Indexed Transcript:\n\n{indexed_transcript}")
    ])
    
    # Step 4: Connect the prompt template to the structured AI model and run it
    chain = prompt | structured_llm
    result: SummarizerOutput = chain.invoke({"indexed_transcript": state["indexed_transcript"]})
    
    # Step 5: Save the AI's summary, decisions list, and activity log entry to the shared clipboard
    action_label = "Correct Summary & Decisions" if critique_prompt_addition else "Summarize & Extract Decisions"
    return {
        "summary": result.summary,
        "decisions": result.decisions,
        "activity_log": [
            AgentLog(
                agent_name="SummarizerAgent",
                action=action_label,
                details=f"Generated summary and identified {len(result.decisions)} key decision(s)."
            )
        ]
    }


# ==============================================================================
# NODE 3: Action Extractor Agent (To-Do Task Finder)
# ==============================================================================
# This worker scans the numbered transcript for action items (tasks to be done).
# - It reads speaker labels (e.g., "Ravi: I will complete API testing by Friday")
#   to assign "Ravi" as the owner and "Friday" as the deadline.
# - It keeps the exact deadline text from the transcript (never inventing calendar dates).
# - If an owner or deadline is missing or vague (e.g., "someone"), it leaves it
#   empty (None) and marks `clarification_status` as "Unresolved" so we can ask the user.
def action_extractor_node(state: MeetingState) -> dict:
    """Extracts action items, capturing task, owner, and deadline, identifying missing data."""
    # Step 1: Force the AI to return its output in the 'ActionExtractorOutput' format
    structured_llm = llm.with_structured_output(ActionExtractorOutput)
    
    # Step 2: Check if we are re-running this step because the Reviewer AI requested a correction.
    critique_prompt_addition = ""
    if state.get("review_feedback") and not state["review_feedback"].is_valid:
        critique_prompt_addition = (
            f"\nIMPORTANT - PREVIOUS REVIEWER CORRECTION REQUEST TO FIX:\n"
            f"{state['review_feedback'].feedback_notes}\n"
        )

    # Step 3: Build instructions telling the AI how to identify tasks without guessing
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an Action Item Extraction Agent. Analyze the indexed transcript and identify all action items (tasks).\n"
         "For each action item:\n"
         "- id: Assign a sequential integer id (1, 2, 3...).\n"
         "- task: Clear, concise description of the task to be performed.\n"
         "- owner: Use the speaker's name if they commit to the task (e.g., 'Ravi: I will test the API' -> owner='Ravi'). "
         "Set to null (None) if unassigned or vague (e.g., 'someone needs to...').\n"
         "- deadline: Copy the EXACT deadline phrase stated in the transcript (e.g., 'Friday', 'Monday 5 PM', 'tonight 8 PM', 'Wednesday noon'). "
         "NEVER convert relative days into invented calendar/ISO dates (do NOT output dates like '2023-10-02'). "
         "Set to null (None) if no deadline is mentioned.\n"
         "- line_reference: Exact line label (e.g., L02) where the task appears.\n"
         "Do NOT guess or fabricate missing owners or deadlines. Leave missing values as null.\n"
         f"{critique_prompt_addition}"),
        ("user", "Indexed Transcript:\n\n{indexed_transcript}")
    ])
    
    # Step 4: Connect the prompt to the AI and run it
    chain = prompt | structured_llm
    result: ActionExtractorOutput = chain.invoke({"indexed_transcript": state["indexed_transcript"]})
    
    # Step 5: Tag initial clarification status for each extracted action item.
    # If either owner or deadline is missing, mark 'clarification_status' as "Unresolved"
    # so the workflow knows to ask the human user for clarification.
    for item in result.action_items:
        has_owner = bool(item.owner and item.owner.strip().lower() not in ["none", "null", "unassigned", "unresolved"])
        has_deadline = bool(item.deadline and item.deadline.strip().lower() not in ["none", "null", "unspecified", "unresolved"])
        
        if not has_owner:
            item.owner = None
        if not has_deadline:
            item.deadline = None
            
        if not has_owner or not has_deadline:
            item.clarification_status = "Unresolved"
        else:
            item.clarification_status = "Not Needed"
            
    # Step 6: Write the extracted action items and log entry back to the shared clipboard
    action_label = "Correct Action Items" if critique_prompt_addition else "Extract Action Items"
    return {
        "action_items": result.action_items,
        "activity_log": [
            AgentLog(
                agent_name="ActionExtractorAgent",
                action=action_label,
                details=f"Extracted {len(result.action_items)} action item(s)."
            )
        ]
    }


# ==============================================================================
# NODE 4: Reviewer Agent (Quality Control & Fact-Checker)
# ==============================================================================
# This AI acts like a strict auditor BEFORE we present or ask the user for final
# inputs. It compares the numbered transcript against the AI-extracted summary,
# decisions, and action items to make sure nothing was hallucinated (made up).
# - If it finds any unsupported statement, it flags it (is_valid=False) and writes
#   specific correction instructions in `feedback_notes`.
def reviewer_node(state: MeetingState) -> dict:
    """Checks the summary, decisions, and action items against the transcript, flags unsupported statements, and requests corrections."""
    # Step 1: Force the AI to output a 'ReviewFeedback' object (is_valid: True/False + feedback_notes)
    structured_llm = llm.with_structured_output(ReviewFeedback)
    
    # Step 2: Track how many review passes have run so far (used to cap correction to 1 pass)
    current_retry = state.get("retry_count", 0)
    
    # Step 3: Format the decisions and action items into readable text lists for the Reviewer AI
    decisions_text = "\n".join([f"- [{d.line_reference}] {d.decision}" for d in state["decisions"]]) or "None"
    actions_text = "\n".join([
        f"- ID {a.id}: Task='{a.task}', Owner='{a.owner}', Deadline='{a.deadline}', "
        f"ClarificationStatus='{a.clarification_status}', Ref='{a.line_reference}'" 
        for a in state["action_items"]
    ]) or "None"
    
    # Step 4: Build the fact-checking prompt with clear validation rules
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Quality & Verification Reviewer Agent. Fact-check the extracted summary, decisions, and action items against the transcript.\n"
         "CRITICAL VALIDATION RULES:\n"
         "1. Summary check: The summary is a concise overview; do NOT mark is_valid=False just because it omits minor details. Only flag the summary if it makes a false claim that contradicts the transcript.\n"
         "2. 'Someone' = Unassigned (Owner MUST be 'None'): If a speaker says 'We need someone to...' or 'Someone should...' (e.g., 'Rohan: someone should notify...'), the task is UNASSIGNED. Owner MUST be 'None' (not the speaker!). Having Owner='None' or Deadline='None' is 100% VALID because a downstream Human Clarification node will ask the user.\n"
         "3. Direct commitment ('I will...'): If a speaker says 'Name: I will do X', then Owner='Name' is VALID.\n"
         "4. Line references: Every cited line reference (e.g., L04) must match the line where that decision or task appears.\n"
         "5. When to set is_valid=False: ONLY set is_valid=False if an item is fabricated/hallucinated, if a 'someone' task was wrongly given an owner instead of 'None', if an invented calendar date (like '2023-10-02') was used, or if a line reference is wrong.\n"
         "6. Otherwise, set is_valid=True and state 'All extracted summary points, decisions, and action items are supported by the transcript.'"),
        ("user",
         "ORIGINAL INDEXED TRANSCRIPT:\n{indexed_transcript}\n\n"
         "EXTRACTED SUMMARY:\n{summary}\n\n"
         "EXTRACTED DECISIONS:\n{decisions_text}\n\n"
         "EXTRACTED ACTION ITEMS:\n{actions_text}")
    ])
    
    # Step 5: Connect prompt and AI model, then run the fact-check
    chain = prompt | structured_llm
    feedback: ReviewFeedback = chain.invoke({
        "indexed_transcript": state["indexed_transcript"],
        "summary": state["summary"],
        "decisions_text": decisions_text,
        "actions_text": actions_text
    })
    
    # Step 6: Save the Reviewer's verdict, update retry_count, and log the review outcome
    note_summary = feedback.feedback_notes if feedback.feedback_notes else ("All statements verified against transcript." if feedback.is_valid else "Unsupported statements flagged for correction.")
    return {
        "review_feedback": feedback,
        "retry_count": current_retry + 1,
        "activity_log": [
            AgentLog(
                agent_name="ReviewerAgent",
                action="Fact Check & Verification",
                details=f"Valid={feedback.is_valid}. Notes: {note_summary}"
            )
        ]
    }


# ==============================================================================
# NODE 5: Human-in-the-Loop Clarification (Interactive Q&A with User)
# ==============================================================================
# Requirement from specification:
# "When an action item has no clear owner or deadline, ask the user for clarification
#  rather than inventing information. Incorporate the response into the report.
#  If the user cannot clarify, mark the field as unresolved."
def human_clarification_node(state: MeetingState) -> dict:
    """CLI-based human-in-the-loop prompt to clarify unassigned tasks or missing deadlines."""
    updated_items = []
    logs = []
    
    print("\n" + "=" * 65)
    print(" [HUMAN-IN-THE-LOOP] Missing Information Detected")
    print(" Please provide missing details below (or press Enter to leave Unresolved).")
    print("=" * 65)
    
    # Step 1: Loop through every action item and check if owner or deadline is missing
    for item in state["action_items"]:
        needs_owner = not item.owner or item.owner.strip().lower() in ["none", "null", "unassigned", "unresolved"]
        needs_deadline = not item.deadline or item.deadline.strip().lower() in ["none", "null", "unspecified", "unresolved"]
        
        # Step 2: If either owner or deadline is missing, ask the user in the terminal
        if needs_owner or needs_deadline:
            print(f"\nTask ID {item.id}: '{item.task}' (Transcript Ref: {item.line_reference})")
            
            # Ask human for the missing owner
            new_owner = item.owner
            if needs_owner:
                val = input("  -> Owner is missing. Enter owner name (or press Enter to mark as Unresolved): ").strip()
                new_owner = val if val else None
            
            # Ask human for the missing deadline
            new_deadline = item.deadline
            if needs_deadline:
                val = input("  -> Deadline is missing. Enter deadline (or press Enter to mark as Unresolved): ").strip()
                new_deadline = val if val else None
            
            # Step 3: Update the task fields and clarification status based on the user's answer:
            # - If the user left any missing field blank, mark that field as "Unresolved"
            #   and set clarification_status = "Unresolved".
            # - If the user answered all missing fields for this task, set clarification_status = "Resolved".
            if (needs_owner and not new_owner) or (needs_deadline and not new_deadline):
                item.clarification_status = "Unresolved"
            else:
                item.clarification_status = "Resolved"
                
            item.owner = new_owner if new_owner else "Unresolved"
            item.deadline = new_deadline if new_deadline else "Unresolved"
            
            # Step 4: Record what the human answered in the audit log
            logs.append(AgentLog(
                agent_name="HumanClarification",
                action="Ask User for Clarification",
                details=f"Task {item.id} ({item.line_reference}) -> Owner: '{item.owner}', Deadline: '{item.deadline}', Status: {item.clarification_status}"
            ))
        else:
            # Both owner and deadline were already clearly stated in the transcript
            item.clarification_status = "Not Needed"
            
        updated_items.append(item)
        
    print("=" * 65 + "\n")
    
    # Step 5: Save the updated action items and human clarification logs to the shared clipboard
    return {
        "action_items": updated_items,
        "activity_log": logs
    }


# ==============================================================================
# NODE 6: Report Assembler (Final Publisher)
# ==============================================================================
# Requirement from specification:
# "Expected Output: Produce a meeting summary, a list of decisions, an action-item
#  table containing tasks, owners, deadlines, and clarification status, and a brief
#  agent activity log."
#
# This final worker takes all verified data from the shared clipboard, formats it
# into clean Markdown sections and tables, and writes:
#   1. `meeting_report.md` (The complete reviewed meeting report including the brief activity log)
#   2. `agent_activity_log.txt` (A standalone audit log file)
def report_assembler_node(state: MeetingState) -> dict:
    """Assembles the final reviewed meeting report and writes meeting_report.md and agent_activity_log.txt."""
    # Step 1: Get current date & time string for the report header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Step 2: Build Section 1 - Executive Summary
    summary_section = f"## 1. Executive Summary\n{state['summary']}\n"
    
    # Step 3: Build Section 2 - Key Decisions (with transcript line references)
    decisions_data = [[d.line_reference, d.decision] for d in state["decisions"]]
    decisions_table = tabulate(decisions_data, headers=["Line Ref", "Decision"], tablefmt="github")
    decisions_section = f"## 2. Key Decisions\n{decisions_table}\n"
    
    # Step 4: Build Section 3 - Action Items Table
    # Contains: ID, Task Description, Owner, Deadline, Clarification Status, and Line Reference
    action_data = [
        [
            a.id,
            a.task,
            a.owner if a.owner else "Unresolved",
            a.deadline if a.deadline else "Unresolved",
            a.clarification_status,
            a.line_reference
        ]
        for a in state["action_items"]
    ]
    actions_table = tabulate(
        action_data, 
        headers=["ID", "Task Description", "Owner", "Deadline", "Clarification Status", "Line Ref"], 
        tablefmt="github"
    )
    actions_section = f"## 3. Action Items\n{actions_table}\n"

    # Step 5: Build Section 4 - Review & Fact-Check Summary
    feedback = state.get("review_feedback")
    if feedback:
        review_status = "Verified (All statements supported by transcript)" if feedback.is_valid else "Flagged (Corrections requested)"
        review_notes = feedback.feedback_notes or "All extracted summary points, decisions, and action items match the transcript."
    else:
        review_status = "Not Run"
        review_notes = "N/A"
    review_section = (
        f"## 4. Review & Verification Summary\n"
        f"- **Verification Status:** {review_status}\n"
        f"- **Reviewer Notes:** {review_notes}\n"
    )

    # Step 6: Build Section 5 - Brief Agent Activity Log
    # Append this final ReportAssembler step to the list of logs collected so far
    final_log_entry = AgentLog(
        agent_name="ReportAssembler",
        action="Assemble & Export Final Report",
        details="Generated reviewed meeting report (meeting_report.md) and activity log (agent_activity_log.txt)."
    )
    current_logs = state["activity_log"] + [final_log_entry]

    activity_table_data = [
        [idx + 1, log.agent_name, log.action, log.details]
        for idx, log in enumerate(current_logs)
    ]
    activity_table = tabulate(
        activity_table_data,
        headers=["Step", "Agent Name", "Task Performed", "Details"],
        tablefmt="github"
    )
    activity_section = f"## 5. Brief Agent Activity Log\n{activity_table}\n"
    
    # Step 7: Combine all 5 sections into the final Markdown report
    meeting_report_content = (
        f"# Reviewed Meeting Report\n"
        f"*Generated on: {timestamp}*\n\n"
        f"{summary_section}\n"
        f"{decisions_section}\n"
        f"{actions_section}\n"
        f"{review_section}\n"
        f"{activity_section}"
    )

    # Step 8: Save `meeting_report.md` to disk
    with open("meeting_report.md", "w", encoding="utf-8") as f:
        f.write(meeting_report_content)

    # Step 9: Format and save `agent_activity_log.txt` to disk as a standalone text log
    log_entries = [
        f"[Step {idx + 1}] [{timestamp}] [{log.agent_name}] -> {log.action}\n    Details: {log.details}"
        for idx, log in enumerate(current_logs)
    ]
    log_file_content = (
        f"======================================================================\n"
        f" MULTI-AGENT ACTIVITY AUDIT LOG\n"
        f" Generated: {timestamp}\n"
        f" Total Lifecycle Events: {len(current_logs)}\n"
        f"======================================================================\n\n"
        + "\n\n".join(log_entries)
        + "\n\n========================= END OF AUDIT LOG =========================\n"
    )

    with open("agent_activity_log.txt", "w", encoding="utf-8") as f:
        f.write(log_file_content)

    print("\n[Artifacts Created]")
    print(" -> File saved: meeting_report.md")
    print(" -> File saved: agent_activity_log.txt")

    # Step 10: Return the final report string and final log entry to the shared clipboard
    return {
        "final_report": meeting_report_content,
        "activity_log": [final_log_entry]
    }
