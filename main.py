

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import requests
import json
from openai import OpenAI
from colorama import Fore, Style, init
import time
import pyfiglet
from dotenv import dotenv_values

init(autoreset=True)

# Avoid persistence between sessions of .env loaded variables
config = dotenv_values(".env")

EXA_API_KEY = config.get("EXA_API_KEY") or os.getenv("EXA_API_KEY")
OPENAI_API_KEY = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = config.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = config.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4-turbo"
EXA_BASE_URL = "https://api.exa.ai"

openai_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

sys_prompt = """
You are a highly capable AI research expert operating like a junior analyst in venture capital, consulting, or investment banking. Your role involves comprehensive market research, competitor analysis, and qualitative studies. You excel in understanding complex queries, reasoning through challenges, and strategic planning.

Key Capabilities:
- **Internet Research:** Access, evaluate, and extract insights from a wide range of credible sources.
- **Data Synthesis:** Transform diverse information into coherent, well-organized, and actionable insights.
- **Iterative Refinement:** Continuously update your context by revising your plan and performing additional research as needed.
- **Structured Output Generation:** When initiating a search, generate a structured JSON output to call the `exa_search` function.

Workflow:
1. **Plan Creation:** Analyze the user’s query and develop a structured research plan.
2. **Generate Search Query:** Formulate precise search queries using your available tools.Generate queries corresponding to year 2024.
3. **Structured Output for exa_search:** For each search action, output a JSON structure.
4. **Execute and Iterate:** Use the generated query to call `exa_search`, review the returned information, update your context, and refine your plan.
5. **Final Reporting:** Synthesize all gathered information into a well-formatted , in depth detailed, actionable report.

Maintain clarity, accuracy, and efficiency in every step of the process. in the text you'll get from exa search , retain the inline references and use them in final synthesiezed answer.
"""

def exa_search(query: str) -> dict:
    """Execute search using Exa API with citations"""
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    payload = {"query": query, "stream" : False, "text": True, "mode": "fast"}
    response = requests.post(f"{EXA_BASE_URL}/search", json=payload, headers=headers)
    return response.json()

def generate_research_step(user_query: str, context: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    """Generate next research step using OpenAI with structured output"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_query}
    ]
    
    #previous context as assistant messages
    for step in context:
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "action": "exa_search",
                "query": step["query"],
                "reasoning": step["reasoning"],
                "plan": step["plan"]
            })
        })
        messages.append({
            "role": "system",
            "content": f"Search Results from '{step['query']}':\n{step['results']}"
        })
    
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}


def process_search_results(results: dict) -> str:
    """Process and format Exa search results"""
    if not results.get("results"):
        return "No relevant results found"
    
    summary = []
    for result in results["results"]:
        summary.append(f"### {result.get('title', 'Untitled')}")
        summary.append(f"**URL**: {result.get('url')}")
        summary.append(f"**Content**: {result.get('text')}...")
        if result.get("highlights"):
            summary.append(f"**Highlights**:\n- " + "\n- ".join(result["highlights"]))
        summary.append("\n")
    
    return "\n".join(summary)

def generate_final_report(context: List[Dict[str, Any]]) -> str:
    """Generate final synthesized report using OpenAI"""
    research_history = "\n\n".join(
        f"## Research Step {i+1}\n"
        f"**Query**: {step['query']}\n"
        f"**Results**:\n{step['results']}"
        for i, step in enumerate(context)
    )
    
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt + "\n\nMaintain the inline citations provided in the data and use them in final answer as inline citations. give output in md and Synthesize the final report using the following research data:"},
            {"role": "user", "content": research_history}
        ],
        temperature=0.5,
        # stream=True,
    )
    
    return response.choices[0].message.content

class ResearchAgent:
    def __init__(self):
        self.context = []
        self.max_iterations = 9  

    def execute_research_plan(self, user_query: str):
        """Execute iterative research process"""
        print(f"{Fore.YELLOW}Initializing research plan...{Style.RESET_ALL}")
        
        for i in range(self.max_iterations):
            print(f"{Fore.CYAN}\n=== next step ==={Style.RESET_ALL}")
            
            research_step = generate_research_step(user_query, self.context)
            
            print(f"{Fore.MAGENTA}\nReasoning:{Style.RESET_ALL}")
            print(research_step.get("reasoning"))
            
     
            
            if not research_step.get("query"):
                print(f"{Fore.RED}No search query generated{Style.RESET_ALL}")
                break
                
            # Execute search
            print(f"{Fore.YELLOW}\nExecuting search: {research_step['query']}{Style.RESET_ALL}")
            search_results = exa_search(research_step["query"])
            processed_results = process_search_results(search_results)
            
            # Store context
            self.context.append({
                "query": research_step["query"],
                "reasoning": research_step.get("reasoning", ""),
                "plan": research_step.get("plan", ""),
                "results": processed_results
            })
            
            # Print search results 
            print(f"{Fore.GREEN}\nSearch Results Summary:{Style.RESET_ALL}")
            print(processed_results[:500] + "...")
            
            # Check completion condition
            if "no relevant results" in processed_results.lower():
                print(f"{Fore.RED}Terminating due to lack of results{Style.RESET_ALL}")
                break
        
      
        print(f"{Fore.GREEN}\nGenerating final report...{Style.RESET_ALL}")
        return generate_final_report(self.context)

def display_banner():
    banner = pyfiglet.figlet_format("Deeper Seeker v1")
    print(Fore.CYAN + banner + Style.RESET_ALL)

if __name__ == "__main__":
    display_banner()

    agent = ResearchAgent()
    user_input = input("Enter your research query: ")
    final_report = agent.execute_research_plan(user_input)
    
    print(f"\n{Fore.GREEN}=== FINAL REPORT ==={Style.RESET_ALL}")
    print(final_report)
    print(f"\n{Fore.CYAN}=== RESEARCH CONTEXT ==={Style.RESET_ALL}")
    print(json.dumps(agent.context, indent=2))