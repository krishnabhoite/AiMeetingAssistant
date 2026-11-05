# ==============================================================================
# FILE OVERVIEW: state.py (Data Blueprints & Shared Memory)
# ==============================================================================
# Think of this file as the "Shared Clipboard" or "Filing Cabinet" for our AI
# assistants. When multiple AI workers (agents) collaborate on a meeting transcript,
# they need a strict, agreed-upon format to write down their findings and pass
# information to the next worker in line.
#
# Key Concepts Explained for a Layman:
# 1. Pydantic (BaseModel & Field):
#    Normally, AI models reply with free-form chat text. Pydantic lets us define
#    strict "forms" (templates) that force the AI to reply with structured data
#    (like filling out specific boxes on a form instead of writing an essay).
# 2. TypedDict (MeetingState):
#    In LangGraph, the "State" is a single shared dictionary (like a digital folder)
#    that gets passed from one step (node) to the next. Every step reads what it
#    needs from this folder and writes its new results back into it.
# ==============================================================================

import operator  # Built-in Python tool; used here to combine (append) lists together
from typing import Annotated, List, Literal, Optional  # Tools to specify data types (text, list, optional, etc.)
from typing_extensions import TypedDict  # Used to define a dictionary with a fixed set of keys and value types
from pydantic import BaseModel, Field  # Library that forces AI outputs to follow strict data structures


# ------------------------------------------------------------------------------
# 1. Blueprint for a Single Meeting Decision
# ------------------------------------------------------------------------------
class Decision(BaseModel):
    # 'Field(description=...)' acts as an instruction label directly for the AI,
    # telling it exactly what kind of information to put into each field.
    decision: str = Field(description="Summary of the decision reached")
    line_reference: str = Field(description="Exact transcript line identifier, e.g., 'L04'")


# ------------------------------------------------------------------------------
# 2. Blueprint for a Single Action Item (To-Do Task)
# ------------------------------------------------------------------------------
class ActionItem(BaseModel):
    # A unique number for each task (1, 2, 3...)
    id: int = Field(description="Unique incremental integer identifier for the task")
    
    # What needs to be done
    task: str = Field(description="Concise description of the task")
    
    # 'Optional[str]' means this can either be text (a person's name) OR empty (None)
    # if nobody was assigned to the task during the meeting.
    owner: Optional[str] = Field(
        default=None, 
        description="Name of the person assigned to the task (use speaker name if they say 'I will...'). Null if unassigned or vague like 'someone'."
    )
    
    # When the task is due (or None if no deadline was mentioned in the meeting)
    deadline: Optional[str] = Field(
        default=None, 
        description="Exact deadline phrase stated in the transcript (e.g., 'Friday', 'Monday 5 PM', 'tonight 8 PM'). Do NOT invent calendar dates. Null if unspecified."
    )
    
    # 'Literal[...]' restricts this field to ONLY one of these three exact phrases.
    # It tracks whether a human needed to step in to fill in missing owner/deadline info.
    clarification_status: Literal["Resolved", "Unresolved", "Not Needed"] = Field(
        default="Not Needed",
        description="Clarification state: 'Not Needed' if fully defined, 'Resolved' if provided by human, 'Unresolved' if left blank."
    )
    
    # Which line of the meeting transcript this task came from (e.g., 'L02')
    line_reference: str = Field(description="Line identifier from which this task was derived, e.g., 'L02'")


# ------------------------------------------------------------------------------
# 3. Blueprint for the Summarizer AI's Output
# ------------------------------------------------------------------------------
# When the Summarizer AI finishes reading the transcript, it must return its answer
# packaged in this exact structure: one summary paragraph + a list of Decisions.
class SummarizerOutput(BaseModel):
    summary: str = Field(description="High-level overview of the meeting's discussion points")
    decisions: List[Decision] = Field(description="List of key decisions made with line references")


# ------------------------------------------------------------------------------
# 4. Blueprint for the Action Extractor AI's Output
# ------------------------------------------------------------------------------
# When the Action Extractor AI finishes, it must return a list of ActionItem objects.
class ActionExtractorOutput(BaseModel):
    action_items: List[ActionItem] = Field(description="List of all actionable tasks identified")


# ------------------------------------------------------------------------------
# 5. Blueprint for the Fact-Checker (Reviewer) AI's Output
# ------------------------------------------------------------------------------
# The Reviewer AI checks if the other AIs made anything up ("hallucinated").
# It returns True/False (is_valid) and notes explaining what needs to be fixed.
class ReviewFeedback(BaseModel):
    is_valid: bool = Field(
        description="True if all extracted summary claims, decisions, and tasks are supported by the transcript without fabricated owners, dates, or facts."
    )
    feedback_notes: str = Field(
        default="", 
        description="If is_valid is False, specify only the fabricated/unsupported statements to fix. If is_valid is True, briefly confirm verification."
    )


# ------------------------------------------------------------------------------
# 6. Blueprint for an Audit Log Entry
# ------------------------------------------------------------------------------
# Every time an AI worker finishes a step, it creates an AgentLog record so we have
# a complete history (audit trail) of who did what during the program's run.
class AgentLog(BaseModel):
    agent_name: str  # Name of the worker/step (e.g., "SummarizerAgent")
    action: str      # Short title of what it did (e.g., "Summarize & Extract Decisions")
    details: str     # Extra details (e.g., "Generated summary and identified 2 decisions.")


# ------------------------------------------------------------------------------
# 7. The Master Shared Clipboard (LangGraph State)
# ------------------------------------------------------------------------------
# 'MeetingState' is the central memory shared across the entire LangGraph workflow.
# Think of it as a packet of papers passed from desk to desk in an office:
# - Desk 1 reads 'raw_transcript' and writes 'indexed_transcript'.
# - Desk 2 reads 'indexed_transcript' and writes 'summary' and 'decisions'.
# - And so on, until the final report is assembled!
class MeetingState(TypedDict):
    raw_transcript: str       # The original, unedited meeting text provided at the start
    indexed_transcript: str   # The meeting text with line numbers added (L01, L02, etc.)
    
    # Core outputs
    summary: str                   # High-level summary of the meeting
    decisions: List[Decision]      # List of decisions agreed upon in the meeting
    action_items: List[ActionItem] # List of to-do tasks found in the meeting
    
    # Review & Loop management
    review_feedback: Optional[ReviewFeedback]  # Pass/Fail result and notes from the Reviewer AI
    retry_count: int                           # Counter to prevent the AI from looping forever if it keeps failing review
    
    # Final markdown deliverable
    final_report: str              # The complete formatted report text
    
    # Audit log (operator.add appends logs across nodes)
    # LAYMAN NOTE on 'Annotated[List[AgentLog], operator.add]':
    # Normally, when a step updates a field in MeetingState, it overwrites the old value.
    # By attaching 'operator.add', we tell LangGraph: "Do NOT overwrite the old logs!
    # Instead, ADD (append) the new log entries to the existing list of logs."
    activity_log: Annotated[List[AgentLog], operator.add]
