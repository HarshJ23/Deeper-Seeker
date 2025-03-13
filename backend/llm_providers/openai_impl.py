from openai import OpenAI
from typing import Dict
import json

from prompts import FOLLOWUP_PROMPT, RESEARCH_PLAN_PROMPT, GEN_QUERY_PROMPT, GEN_REPORT_PROMPT
from json_extraction import extract_json_from_response

# OpenAI/Groq implementations
def generate_followup(client: OpenAI, model: str, context: str) -> Dict:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FOLLOWUP_PROMPT},
            {"role": "user", "content": context}
        ],
        response_format={"type": "json_object"},
    )
    return extract_json_from_response(completion.choices[0].message.content)

def generate_research_plan(client: OpenAI, model: str, initial_query: str, followup_context: str) -> Dict:
    combined_context = initial_query + followup_context
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RESEARCH_PLAN_PROMPT},
            {"role": "user", "content": combined_context}
        ],
        response_format={"type": "json_object"},
    )
    return extract_json_from_response(completion.choices[0].message.content)

def generate_queries_for_step(client: OpenAI, model: str, step: str, description: str) -> Dict:
    # Implementation here
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GEN_QUERY_PROMPT},
                {"role": "user", "content": description}
            ],
            response_format={"type": "json_object"},
        )
        queries = extract_json_from_response(completion.choices[0].message.content)
        # Write the generated queries to the output file
        # write_output_to_file(f"### Generated Queries for {step}\n" + json.dumps(queries, indent=2))
        return {step: queries}
    except json.JSONDecodeError:
        return {step: {"error": "Failed to generate queries"}}

def generate_report(client: OpenAI, model: str, prompt: str, learnings: str) -> str:
    """Generate a detailed markdown report using OpenAI."""
    try:
        # Format the prompt with the user's query and research learnings
        formatted_prompt = GEN_REPORT_PROMPT.format(
            prompt=prompt,
            learnings=learnings
        )
        
        # Call the OpenAI API
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional research analyst."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.5,  # Adjust for creativity vs. consistency
        )
        
        # Extract and return the report content
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating report with OpenAI: {e}")
        return f"Error generating report: {e}"