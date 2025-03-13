from typing import Dict, Callable, Any
import llm_providers.google_genai_impl as google_genai_impl
import llm_providers.openai_impl as openai_impl

# We do not set the types of the callable since the code structure is not yet stable

GENERATE_FOLLOWUP: Dict[str, Callable[..., Any]] = {
    "openai": openai_impl.generate_followup,
    "groq": openai_impl.generate_followup,
    "gemini": google_genai_impl.generate_followup
}

GENERATE_RESEARCH_PLAN: Dict[str, Callable[..., Any]] = {
    "openai": openai_impl.generate_research_plan,
    "groq": openai_impl.generate_research_plan,
    "gemini": google_genai_impl.generate_research_plan
}

GENERATE_QUERIES_FOR_STEP: Dict[str, Callable[..., Any]] = {
    "openai": openai_impl.generate_queries_for_step,
    "groq": openai_impl.generate_queries_for_step,
    "gemini": google_genai_impl.generate_queries_for_step
}

GENERATE_REPORT: Dict[str, Callable[..., Any]] = {
    "openai": openai_impl.generate_report,
    "groq": openai_impl.generate_report,
    "gemini": google_genai_impl.generate_report
}

