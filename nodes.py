# ==============================================================================
# FILE OVERVIEW: nodes.py (The AI Workers / Stations on the Assembly Line)
# ==============================================================================
# In LangGraph, a "Node" is simply a Python function that performs one specific
# job. Imagine an office assembly line where each desk has a specialist:
#   1. Preprocessor Node: Adds line numbers (L01, L02...) to the raw transcript.
#   2. Summarizer Node: An AI that writes an executive summary & lists decisions.
#   3. Action Extractor Node: An AI that finds to-do tasks, owners, and deadlines.
#   4. Human Clarification Node: Pauses and asks a real human (via keyboard input)
#      to fill in any missing task owners or deadlines.
#   5. Reviewer Node: A "Fact-Checker" AI that verifies no information was made up.
#   6. Report Assembler Node: Formats everything into neat tables and saves files.
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
from tabulate import tabulate  # (Duplicate import kept to preserve exact original code)
from state import MeetingState, AgentLog  # Importing our shared state and log blueprints

from state import (
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
# LLM initialized via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPEN_API_BASE"),
    temperature=0.0
)


# ==============================================================================
# NODE 1: Preprocessor (Line Numbering Worker)
# ==============================================================================
# Why do we need this?
# If we ask an AI to cite where a decision came from, it's much easier and more
# accurate if every line already has an ID tag like "L01:", "L02:", etc.
# Notice this node doesn't even use AI—it just uses simple Python text formatting!
def preprocessor_node(state: MeetingState) -> dict:
    """Indexes each line of the raw transcript with line numbers (L01, L02, etc.)."""
    # Split the raw transcript by line breaks, remove extra spaces, and ignore empty lines
    lines = [line.strip() for line in state["raw_transcript"].strip().split("\n") if line.strip()]
    
    # Attach a two-digit line number (L01, L02, L03...) to the front of each line
    indexed_lines = [f"L{idx+1:02d}: {line}" for idx, line in enumerate(lines)]
    
    # Join all the numbered lines back into one single block of text
    indexed_transcript = "\n".join(indexed_lines)
    
    # Return updates to the shared clipboard (MeetingState):
    # - Save the newly numbered transcript
    # - Reset retry_count to 0
    # - Add an entry to the activity_log recording that Preprocessor finished its job
    return {
        "indexed_transcript": indexed_transcript,
        "retry_count": 0,
        "activity_log": [
            AgentLog(
                agent_name="Preprocessor",
                action="Index Transcript",
                details=f"Numbered {len(lines)} lines of text."
            )
        ]
    }


# ==============================================================================
# NODE 2: Summarizer Agent (Summary & Decision Extractor)
# ==============================================================================
# This worker reads the numbered transcript and extracts:
#   1. A high-level executive summary of the meeting.
#   2. A list of concrete decisions made, along with the line number (e.g., L01).
def summarizer_node(state: MeetingState) -> dict:
    """Extracts the high-level meeting summary and decisions with line citations."""
    # Tell the AI that its answer MUST match the 'SummarizerOutput' form from state.py
    structured_llm = llm.with_structured_output(SummarizerOutput)
    
    # Create the instructions for the AI:
    # - "system" message sets the AI's role and rules.
    # - "user" message provides the actual meeting transcript to analyze.
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a technical meeting summarizer. Given the indexed transcript below, extract:\n"
         "1. A concise overview of discussion topics and context.\n"
         "2. A list of concrete decisions made, with the exact line label (e.g., L03) citing the decision.\n"
         "Do not invent or extrapolate beyond the transcript."),
        ("user", "Indexed Transcript:\n\n{indexed_transcript}")
    ])
    
    # Connect the prompt template to the structured AI model using LangChain's '|' pipe operator
    chain = prompt | structured_llm
    
    # Run the AI chain by passing in the numbered transcript from our shared clipboard
    result: SummarizerOutput = chain.invoke({"indexed_transcript": state["indexed_transcript"]})
    
    # Save the AI's summary, decisions list, and a log entry back to the shared clipboard
    return {
        "summary": result.summary,
        "decisions": result.decisions,
        "activity_log": [
            AgentLog(
                agent_name="SummarizerAgent",
                action="Summarize & Extract Decisions",
                details=f"Generated summary and identified {len(result.decisions)} decisions."
            )
        ]
    }


# ==============================================================================
# NODE 3: Action Extractor Agent (To-Do Task Finder)
# ==============================================================================
# This worker scans the numbered transcript for tasks that need to be done.
# For each task, it tries to find WHO is responsible (owner) and WHEN it's due (deadline).
# If the Reviewer AI previously rejected its work, this node also reads the Reviewer's
# critique and fixes its mistakes!
def action_extractor_node(state: MeetingState) -> dict:
    """Extracts action items, capturing task, owner, and deadline, identifying missing data."""
    # Force the AI to return its output in the 'ActionExtractorOutput' format
    structured_llm = llm.with_structured_output(ActionExtractorOutput)
    
    # Check if we are re-running this step because the Reviewer AI found an error earlier.
    # If so, attach the Reviewer's correction notes to the prompt so the AI learns from its mistake.
    critique_prompt_addition = ""
    if state.get("review_feedback") and not state["review_feedback"].is_valid:
        critique_prompt_addition = f"\nPREVIOUS REVIEWER CRITIQUE TO FIX:\n{state['review_feedback'].feedback_notes}\n"

    # Build the instructions telling the AI how to identify tasks and strictly forbidding guessing
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an Action Item Extraction Agent. Analyze the indexed transcript and identify all explicit or implicit tasks.\n"
         "For each task:\n"
         "- Assign a sequential integer id (1, 2, 3...).\n"
         "- task: Actionable description.\n"
         "- owner: Person's name responsible for it. Set to null if unassigned or ambiguous (e.g., 'someone').\n"
         "- deadline: Exact due date/time if stated. Set to null if unspecified.\n"
         "- line_reference: Exact line label (e.g., L02) where the item appears.\n"
         "Do NOT guess or fabricate missing owners or deadlines. Leave them as null.\n"
         f"{critique_prompt_addition}"),
        ("user", "Indexed Transcript:\n\n{indexed_transcript}")
    ])
    
    # Connect the prompt to the AI and run it
    chain = prompt | structured_llm
    result: ActionExtractorOutput = chain.invoke({"indexed_transcript": state["indexed_transcript"]})
    
    # Tag initial clarification needs:
    # Loop through each extracted task. If either the owner or deadline is missing,
    # mark 'clarification_status' as "Unresolved" so we know to ask a human later.
    for item in result.action_items:
        if not item.owner or not item.deadline:
            item.clarification_status = "Unresolved"
        else:
            item.clarification_status = "Not Needed"
            
    # Write the extracted action items and log entry back to the shared clipboard
    return {
        "action_items": result.action_items,
        "activity_log": [
            AgentLog(
                agent_name="ActionExtractorAgent",
                action="Extract Action Items",
                details=f"Extracted {len(result.action_items)} action items."
            )
        ]
    }


# ==============================================================================
# NODE 4: Human-in-the-Loop Clarification (Interactive Q&A with User)
# ==============================================================================
# Sometimes meetings are vague (e.g., "Someone should review security").
# Instead of letting the AI guess, this node pauses the program and prompts the
# human user in the terminal (`input(...)`) to type in the missing owner or deadline.
def human_clarification_node(state: MeetingState) -> dict:
    """CLI-based human-in-the-loop prompt to clarify unassigned tasks or missing deadlines."""
    updated_items = []
    logs = []
    
    print("\n" + "="*60)
    print(" [HUMAN-IN-THE-LOOP] Missing Information Detected")
    print("="*60)
    
    # Check every action item to see if it is missing an owner or a deadline
    for item in state["action_items"]:
        needs_owner = not item.owner or item.owner.strip().lower() in ["none", "null", "unassigned"]
        needs_deadline = not item.deadline or item.deadline.strip().lower() in ["none", "null", "unspecified"]
        
        # If either owner or deadline is missing, ask the user in the terminal
        if needs_owner or needs_deadline:
            print(f"\nTask ID {item.id}: '{item.task}' (From line: {item.line_reference})")
            
            # Ask human for the missing owner
            new_owner = item.owner
            if needs_owner:
                val = input("  -> Assignee is missing. Enter owner (Leave blank to keep unresolved): ").strip()
                if val:
                    new_owner = val
            
            # Ask human for the missing deadline
            new_deadline = item.deadline
            if needs_deadline:
                val = input("  -> Deadline is missing. Enter deadline (Leave blank to keep unresolved): ").strip()
                if val:
                    new_deadline = val
            
            # Update resolution status:
            # If the human left any required question blank, it remains "Unresolved".
            # If the human provided the missing details, mark it as "Resolved"!
            if (needs_owner and not new_owner) or (needs_deadline and not new_deadline):
                item.clarification_status = "Unresolved"
            else:
                item.clarification_status = "Resolved"
                
            # Default to "Unassigned" or "Unspecified" for clean display if left blank
            item.owner = new_owner if new_owner else "Unassigned"
            item.deadline = new_deadline if new_deadline else "Unspecified"
            
            # Record what the human answered in the audit log
            logs.append(AgentLog(
                agent_name="HumanClarification",
                action="Resolve Ambiguity",
                details=f"Task {item.id} -> Owner: '{item.owner}', Deadline: '{item.deadline}', Status: {item.clarification_status}"
            ))
        else:
            # Both owner and deadline were already present in the transcript
            item.clarification_status = "Not Needed"
            
        updated_items.append(item)
        
    print("="*60 + "\n")
    
    # Save the updated action items and human clarification logs back to the shared clipboard
    return {
        "action_items": updated_items,
        "activity_log": logs
    }


# ==============================================================================
# NODE 5: Reviewer Agent (Quality Control & Fact-Checker)
# ==============================================================================
# This AI acts like a strict auditor. It compares the original numbered transcript
# against the extracted summary, decisions, and action items to make sure nothing
# was hallucinated (made up) or misquoted.
def reviewer_node(state: MeetingState) -> dict:
    """Verifies that all extracted summary items, decisions, and tasks match transcript facts."""
    # Force the AI to output a 'ReviewFeedback' object (is_valid: True/False + feedback_notes)
    structured_llm = llm.with_structured_output(ReviewFeedback)
    
    # Track how many times we have run the review check so far
    current_retry = state.get("retry_count", 0)
    
    # Format the decisions and action items into readable text lists for the Reviewer AI to inspect
    decisions_text = "\n".join([f"- [{d.line_reference}] {d.decision}" for d in state["decisions"]])
    actions_text = "\n".join([
        f"- ID {a.id}: Task='{a.task}', Owner='{a.owner}', Deadline='{a.deadline}', Ref='{a.line_reference}'" 
        for a in state["action_items"]
    ])
    
    # Build the fact-checking prompt with strict validation rules
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Quality & Verification Reviewer Agent. Fact-check the extracted meeting data against the transcript.\n"
         "Validation Rules:\n"
         "1. Every cited line reference must accurately contain the claimed decision or task.\n"
         "2. No owners or deadlines should be fabricated (unless marked as human-resolved).\n"
         "3. If any decision or task completely misrepresents the text, mark is_valid=False and specify precise feedback.\n"
         "4. If all items are supported by the transcript, mark is_valid=True."),
        ("user",
         "ORIGINAL TRANSCRIPT:\n{indexed_transcript}\n\n"
         "EXTRACTED SUMMARY:\n{summary}\n\n"
         "DECISIONS:\n{decisions_text}\n\n"
         "ACTION ITEMS:\n{actions_text}")
    ])
    
    # Connect prompt and AI model, then run the fact-check
    chain = prompt | structured_llm
    feedback: ReviewFeedback = chain.invoke({
        "indexed_transcript": state["indexed_transcript"],
        "summary": state["summary"],
        "decisions_text": decisions_text,
        "actions_text": actions_text
    })
    
    # Save the Reviewer's verdict, increment the retry counter by 1, and log the review outcome
    return {
        "review_feedback": feedback,
        "retry_count": current_retry + 1,
        "activity_log": [
            AgentLog(
                agent_name="ReviewerAgent",
                action="Fact Check & Verification",
                details=f"Review Pass {current_retry + 1}: Valid={feedback.is_valid}. Notes: {feedback.feedback_notes or 'None'}"
            )
        ]
    }


# ==============================================================================
# NODE 6: Report Assembler (Final Publisher)
# ==============================================================================
# Once everything is verified, this final worker takes all the data from the shared
# clipboard, formats it into clean Markdown tables, and saves two files to disk:
#   1. `meeting_report.md` (The polished meeting summary, decisions, and tasks)
#   2. `agent_activity_log.txt` (A step-by-step audit trail of what every agent did)
def report_assembler_node(state: MeetingState) -> dict:
    """Assembles deliverables, prints them, and writes:
       - meeting_report.md (Summary, Decisions, Action Items)
       - agent_activity_log.txt (Chronological agent telemetry)
    """
    # Get current date & time string (e.g., "2026-10-31 20:00:00")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------
    # 1. Build the Meeting Report (Deliverable 1)
    # ---------------------------------------------------------
    summary_section = f"## 1. Executive Summary\n{state['summary']}\n"
    
    # Convert decisions list into a GitHub-flavored Markdown table using 'tabulate'
    decisions_data = [[d.line_reference, d.decision] for d in state["decisions"]]
    decisions_table = tabulate(decisions_data, headers=["Line Ref", "Decision"], tablefmt="github")
    decisions_section = f"## 2. Key Decisions\n{decisions_table}\n"
    
    # Convert action items list into a GitHub-flavored Markdown table
    action_data = [
        [a.id, a.task, a.owner or "Unassigned", a.deadline or "Unspecified", a.clarification_status, a.line_reference]
        for a in state["action_items"]
    ]
    actions_table = tabulate(
        action_data, 
        headers=["ID", "Task Description", "Owner", "Deadline", "Clarification Status", "Reference"], 
        tablefmt="github"
    )
    actions_section = f"## 3. Action Items\n{actions_table}\n"
    
    # Combine all sections into the final Markdown document string
    meeting_report_content = (
        f"# Meeting Review Report\n"
        f"*Generated on: {timestamp}*\n\n"
        f"{summary_section}\n"
        f"{decisions_section}\n"
        f"{actions_section}"
    )

    # Write meeting_report.md to the project folder
    with open("meeting_report.md", "w", encoding="utf-8") as f:
        f.write(meeting_report_content)

    # ---------------------------------------------------------
    # 2. Build the Agent Activity Log File (Deliverable 2)
    # ---------------------------------------------------------
    # Append this final assembly step to the list of logs collected so far
    current_logs = state["activity_log"] + [
        AgentLog(
            agent_name="ReportAssembler",
            action="Assemble & Export",
            details="Exported meeting_report.md and generated agent_activity_log.txt"
        )
    ]

    # Format each log item into a readable text block
    log_entries = [
        f"[{timestamp}] [{log.agent_name}] -> {log.action}\n    Details: {log.details}"
        for log in current_logs
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

    # Write agent_activity_log.txt to the project folder
    with open("agent_activity_log.txt", "w", encoding="utf-8") as f:
        f.write(log_file_content)

    print("\n[Artifacts Created]")
    print(" -> File saved: meeting_report.md")
    print(" -> File saved: agent_activity_log.txt")

    # Save the final report string and final log entry to the shared clipboard
    return {
        "final_report": meeting_report_content,
        "activity_log": [
            AgentLog(
                agent_name="ReportAssembler",
                action="Assemble & Export",
                details="Exported meeting_report.md and agent_activity_log.txt"
            )
        ]
    }
