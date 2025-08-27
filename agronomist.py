import os
import asyncio
from typing import Dict
from dotenv import load_dotenv
import ee  # Google Earth Engine
import requests
from pydantic import BaseModel

from tavily import TavilyClient
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_export_api_key,
    ModelSettings,
    function_tool,
    RunContextWrapper,
    StopAtTools,
    FunctionTool,
    handoff
)
from agents.extensions.handoff_filters import remove_all_tools
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# =============================
# Context: env, clients, model
# =============================
class Context:
    def __init__(self):
        load_dotenv()
        tracing_api_key = os.getenv("OPENAI_API_KEY")
        if tracing_api_key:
            set_tracing_export_api_key(tracing_api_key)

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.external_client = AsyncOpenAI(api_key=self.gemini_api_key, base_url=self.base_url)

        self.model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=self.external_client,
        )

ctx = Context()
ee.Initialize()  # Initialize Earth Engine

# =============================
# Tools
# =============================
@function_tool
async def agri_search(query: str) -> str:
    try:
        results = ctx.tavily.search(query=f"agronomy {query}", max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results or not results.get("results"):
        return "No relevant results found."

    output = "\n🔍 Top agronomy results:\n"
    for idx, r in enumerate(results["results"], 1):
        output += f"{idx}. {r.get('title','No title')} — {r.get('url','')}\n"
    return output

@function_tool
async def crop_monitoring(region: dict, start_date: str, end_date: str, index: str = "NDVI") -> str:
    try:
        polygon = ee.Geometry.Polygon(region["coordinates"])
        collection = (ee.ImageCollection("COPERNICUS/S2")
                      .filterBounds(polygon)
                      .filterDate(start_date, end_date)
                      .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
        ndvi_stats = collection.mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=polygon,
            scale=10
        ).getInfo()
        mean_ndvi = ndvi_stats.get("NDVI", None)
        return f"📊 Mean NDVI for {start_date} to {end_date}: {mean_ndvi:.3f}"
    except Exception as e:
        return f"Crop monitoring failed: {e}"

@function_tool
async def weather_update(city: str) -> str:
    """Get real-time weather data for agronomy decisions using OpenWeatherMap API."""
    try:
        api_key = ctx.weather_api_key
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Weather API error: {data.get('message')}"
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        return (f"🌤 Weather in {city}: {weather_desc}, Temperature: {temp}°C, "
                f"Humidity: {humidity}%, Wind Speed: {wind_speed} m/s")
    except Exception as e:
        return f"Weather update failed: {e}"

def is_user_admin(context: RunContextWrapper, agent: Agent) -> bool:
    return context.get("user_role") == "admin"

@function_tool(is_enabled=is_user_admin)
def delete_field_data() -> str:
    return "✅ Admin tool executed: Field data deleted."

# =============================
# Stateful Tool
# =============================
class AnalysisCounter(FunctionTool):
    def __init__(self):
        self._count = 0
        super().__init__(
            name="analysis_counter",
            description="Counts each crop monitoring or research analysis done.",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=self.on_invoke_tool
        )

    async def on_invoke_tool(self, context, args_json_str) -> str:
        self._count += 1
        return f"📈 Total analyses performed in this session: {self._count}"

counter_tool = AnalysisCounter()

# =============================
# Stateful Instructions
# =============================
class AgriStatefulAgent:
    def __init__(self):
        self.interaction_count = 0
        self.previous_queries = []

    def get_instructions(self, context: RunContextWrapper, agent: Agent) -> str:
        self.interaction_count += 1
        instruction = (
            f"You are {agent.name}, an agronomist assistant. Interaction #{self.interaction_count}.\n"
            "- Break down complex tasks.\n"
            "- Use `agri_search` for research.\n"
            "- Use `crop_monitoring` for NDVI or vegetation analysis.\n"
            "- Use `weather_update` for weather-related guidance.\n"
            "- Cite sources briefly.\n"
        )
        if self.previous_queries:
            instruction += "\nPrevious queries:\n"
            for idx, query in enumerate(self.previous_queries, 1):
                instruction += f"{idx}. {query}\n"
            instruction += "\nBuild on previous conversations when relevant.\n"
        if self.interaction_count == 1:
            instruction = (
                "You are a research assistant for agronomy. "
                "Be welcoming, explain how you can help, and show step-by-step approach."
            )
        return instruction

stateful_agri = AgriStatefulAgent()

# =============================
# Base Agent with Advanced Features
# =============================
base_agent = Agent(
    name="AgroDeepSearchAgent",
    instructions=lambda context, agent: stateful_agri.get_instructions(context, agent),
    tools=[agri_search, crop_monitoring, weather_update, delete_field_data, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.4),
    tool_use_behavior=StopAtTools(stop_at_tool_names=["crop_monitoring", "weather_update"])
)

# =============================
# Specialists
# =============================
soil_agent = base_agent.clone(
    name="SoilExpert",
    instructions="You specialize in soil health and fertility. Provide actionable advice.",
    model_settings=ModelSettings(temperature=0.2)
)

crop_agent = base_agent.clone(
    name="CropExpert",
    instructions="You specialize in crop yield, rotation, NDVI monitoring, and weather impact.",
    model_settings=ModelSettings(temperature=0.3)
)

economics_agent = base_agent.clone(
    name="AgriEconomist",
    instructions="Focus on agricultural economics, market trends, and profitability.",
    model_settings=ModelSettings(temperature=0.2)
)

# =============================
# Advanced Handoff Example
# =============================
class EscalationData(BaseModel):
    reason: str
    field_id: str

async def on_escalation(ctx: RunContextWrapper, input_data: EscalationData):
    print(f"HANDOFF: Escalating field {input_data.field_id} due to {input_data.reason}")

escalation_agent = Agent(name="AgriEscalationAgent")
escalation_handoff = handoff(
    agent=escalation_agent,
    tool_name_override="escalate_to_expert",
    tool_description_override="Escalate complex agronomy issues to a specialist.",
    on_handoff=on_escalation,
    input_type=EscalationData,
    input_filter=remove_all_tools
)

# =============================
# Multi-Agent Coordination
# =============================
async def run_agronomy_team(question: str) -> str:
    synthesis = ""
    # Decide if question requires escalation
    if "urgent" in question.lower() or "problem" in question.lower():
        result = await Runner.run(base_agent, question)
        # Trigger handoff to escalation agent
        handoff_input = EscalationData(reason="Detected complex issue", field_id="Field-001")
        await escalation_handoff.on_handoff(RunContextWrapper({}), handoff_input)
        synthesis += f"\n--- Escalation Handoff Triggered ---\n{result.final_output}\n"
    else:
        for expert in [soil_agent, crop_agent, economics_agent]:
            result = await Runner.run(expert, f"Answer question: {question}", max_turns=5)
            synthesis += f"\n--- {expert.name} ---\n{result.final_output}\n"
    return synthesis

# =============================
# Interactive Loop
# =============================
async def main() -> None:
    print("🌱 Agronomy Deep Research Agent ready! Type 'exit' to quit.\n")
    while True:
        try:
            question = input("❓ Ask your agronomy question: ").strip()
            if question.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break
            if question:
                stateful_agri.previous_queries.append(question)
            print("\n💬 Thinking...\n")
            team_result = await run_agronomy_team(question)
            print(team_result)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break

if __name__ == "__main__":
    asyncio.run(main())
