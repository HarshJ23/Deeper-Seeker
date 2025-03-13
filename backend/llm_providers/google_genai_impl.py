from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Dict

from prompts import FOLLOWUP_PROMPT, RESEARCH_PLAN_PROMPT, GEN_QUERY_PROMPT, GEN_REPORT_PROMPT

class ReportResponse(BaseModel):
    reportMarkdown: str

def generate_followup(client: genai.Client, model: str, context: str) -> Dict:
    # Implementation here
    raise NotImplementedError()

def generate_research_plan(client: genai.Client, model: str, initial_query: str, followup_context: str) -> Dict:
    # Implementation here
    raise NotImplementedError()

def generate_queries_for_step(client: genai.Client, model: str, step: str, description: str) -> Dict:
    # Implementation here
    raise NotImplementedError()

def generate_report(client: genai.Client, model: str, prompt: str, learnings: str) -> str:
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportResponse,
        ),
        contents=[
            f"Given the following prompt from the user, write a final report on the topic using "
            f"the learnings from research. Return a JSON object with a 'reportMarkdown' field "
            f"containing a detailed markdown report (aim for 3+ pages). Include ALL the learnings. use the text field corresponding to the query.The report should be very detailed with inline citations in markdown format :- (text)[source link] in each subsection "
            f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
            f"Here are all the learnings from research:\n\n<learnings>\n{learnings}\n</learnings>"
        ]
    )
    report_response = ReportResponse.parse_raw(response.text)
    return report_response.reportMarkdown