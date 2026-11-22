from smolagents import (
    CodeAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    load_tool,
    tool
)

import litellm

import os
import datetime
from ddgs import DDGS
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

litellm.turn_on_debug()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def search_ai_news(query: str) -> str:
    """
    Search for recent artificial intelligence news.

    Args:
        query: The AI topic to search for.
               Examples: AI agents, LLMs, OpenAI,
               Anthropic, robotics or Hugging Face.

    Returns:
        Recent AI news articles with title, source,
        publication date, description and URL.
    """

    try:
        results = DDGS().news(
            query=f"AI artificial intelligence {query}",
            region="wt-wt",
            safesearch="moderate",
            timelimit="w",
            max_results=5
        )

        if not results:
            return "No recent AI news articles found."

        news_results = []

        for index, article in enumerate(results, start=1):

            news_results.append(
                f"""
                Article {index}
                Title: {article.get('title', 'N/A')}
                Source: {article.get('source', 'N/A')}
                Published: {article.get('date', 'N/A')}
                Description: {article.get('body', 'N/A')}
                URL: {article.get('url', 'N/A')}
                """
            )

        return "\n".join(news_results)

    except Exception as e:
        return f"Error searching AI news: {str(e)}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

search_tool = DuckDuckGoSearchTool(max_results=5)

visit_tool = VisitWebpageTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = LiteLLMModel(
    model_id="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.5,
    max_tokens=2096,
    num_retries=2
)


# Import tool from Hub
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        search_tool,
        visit_tool,
        search_ai_news,
        get_current_time_in_timezone
    ],
    max_steps=6,
    verbosity_level=1,
    planning_interval=None,
    name="ai_news_research_agent",
    description="An agent that searches for recent AI developments.",
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()