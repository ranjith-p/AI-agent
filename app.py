import os
import datetime


import pytz
import yaml
from ddgs import DDGS

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool

from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
import litellm

litellm.turn_on_debug()


@tool
def search_ai_news(query: str, top_n: int = 5, recency_days: int = 7, source: str = "news") -> str:
    """Search for recent artificial intelligence news.

    Args:
        query: The AI topic to search for, e.g. AI agents, LLMs, OpenAI.
        top_n: Maximum number of articles to return.
        recency_days: How recent the articles should be, in days. Kept for
            compatibility; the search uses a one-week window.
        source: Ignored, kept so calls that pass it do not fail.
    """
    try:
        results = DDGS().news(
            query=f"AI artificial intelligence {query}",
            region="wt-wt",
            safesearch="moderate",
            timelimit="w",
            # Enforce in code, not just in the prompt: the model asked for
            # top_n=20 on 2026-11-22 despite 'only query' in the prompt,
            # flooding the context with 20 low-quality articles.
            max_results=min(top_n, 5),
        )
        if not results:
            return "No recent AI news articles found."
        out = []
        for i, a in enumerate(results, start=1):
            out.append(
                f"Article {i}\nTitle: {a.get('title', 'N/A')}\n"
                f"Source: {a.get('source', 'N/A')}\nPublished: {a.get('date', 'N/A')}\n"
                f"Description: {a.get('body', 'N/A')}\nURL: {a.get('url', 'N/A')}"
            )
        return "\n\n".join(out)
    except Exception as e:
        return f"Error searching AI news: {e}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.

    Args:
        timezone: A valid timezone, e.g. 'America/New_York' or 'UTC'.
    """
    try:
        tz = pytz.timezone(timezone)
        return f"The current local time in {timezone} is: {datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {e}"


final_answer = FinalAnswerTool()

# FIX 1: key from environment / Space secret only - never hardcode it.
# FIX 2: temperature 0.5 -> 0.1 (CodeAgent format adherence on Groq)
# FIX 3: max_tokens 2096 -> 8192 (gpt-oss spends tokens on reasoning first;
#         2096 produced the empty code blocks in steps 2, 3, 4 and 6)
model = LiteLLMModel(
    model_id="groq/openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=8192,
    num_retries=2,
)

with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    # FIX 4: dropped VisitWebpageTool - three overlapping search tools
    # (DuckDuckGo, VisitWebpage, search_ai_news) made the model choose
    # badly and invent calls. Two distinct tools are enough for Unit 1.
    tools=[final_answer, DuckDuckGoSearchTool(max_results=5), search_ai_news, get_current_time_in_timezone],
    # FIX 5: 6 steps is too few when 1-2 steps are wasted on a parse error
    max_steps=12,
    verbosity_level=1,
    planning_interval=None,
    name="ai_news_research_agent",
    description="An agent that searches for recent AI developments.",
    prompt_templates=prompt_templates,
)

GradioUI(agent).launch()
