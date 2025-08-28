
import os
import asyncio
from dotenv import load_dotenv
import ee  # Google Earth Engine
from pydantic import BaseModel
from typing import List

ee.Initialize(project='crop-monitoring-project-469812')

from tavily import TavilyClient
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_export_api_key,
    ModelSettings,
    ItemHelpers,
    function_tool,
    RunContextWrapper,
    StopAtTools,
    FunctionTool,
    handoff
)
from agents.extensions.handoff_filters import remove_all_tools
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

class Context:
    def __init__(self):
        load_dotenv()
        tracing_api_key = os.getenv("OPENAI_API_KEY")
        if tracing_api_key:
            set_tracing_export_api_key(tracing_api_key)

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.external_client = AsyncOpenAI(api_key=self.gemini_api_key, base_url=self.base_url)

        self.model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=self.external_client,
        )

ctx = Context()
ee.Initialize()  # Initialize Earth Engine

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


class RegionInput(BaseModel):
    coordinates: List[List[List[float]]]

@function_tool
async def crop_monitoring(region: RegionInput, start_date: str, end_date: str, index: str = "NDVI") -> str:
    try:
        polygon = ee.Geometry.Polygon(region.coordinates)
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

class EscalationData(BaseModel):
    reason: str
    field_id: str

async def on_escalation(ctx: RunContextWrapper, input_data: EscalationData):
    quick_response = f"🚨 Urgent issue detected in field {input_data.field_id}: {input_data.reason}.\n\n"
    quick_response += "✅ Quick advice:\n"
    quick_response += "- Inspect the crop immediately to confirm if pests (like fall armyworm or stem borers) are present.\n"
    quick_response += "- If leaves are yellowing, apply a quick foliar nitrogen spray (e.g., urea 2% solution).\n"
    quick_response += "- Remove and destroy heavily infested leaves if possible.\n"
    quick_response += "- If pest pressure is high, apply a recommended pesticide such as Spinosad or Emamectin Benzoate (follow local guidelines).\n"
    quick_response += "- Ensure adequate irrigation, as stress worsens yellowing.\n"
    quick_response += "\n👉 Would you like to escalate this to a professional agronomist for tailored guidance? (yes/no)"
    return quick_response

escalation_agent = Agent(
    name="AgriEscalationAgent",
    instructions="""
    Handle urgent soil or crop issues (e.g., pests, diseases, crop stress).
    
    - Always suggest a quick and practical immediate solution first.  
    - Then, ask the user: "Would you like to escalate this to a professional agronomist for detailed support?"  
    - If the user says YES → provide the following contact:
        Agronomist: Aleema Saleem
        Phone: +92 123456789
    - If the user says NO → continue giving advice yourself.
    """
)

escalation_handoff = handoff(
    agent=escalation_agent,
    tool_name_override="escalate_to_expert",
    tool_description_override="Escalate complex agronomy issues to a specialist.",
    on_handoff=on_escalation,
    input_type=EscalationData,
    input_filter=remove_all_tools
)

base_agent = Agent(
    name="AgroDeepSearchAgent",
    instructions=lambda context, agent: stateful_agri.get_instructions(context, agent),
    tools=[agri_search, crop_monitoring, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.4),
    tool_use_behavior=StopAtTools(stop_at_tool_names=["crop_monitoring"]),
    handoffs=[escalation_handoff]
)

soil_agent = base_agent.clone(
    name="SoilExpert",
    instructions="You specialize in soil health and fertility. Provide actionable advice.",
    model_settings=ModelSettings(temperature=0.2)
)


crop_agent = base_agent.clone(
    name="CropExpert",
    instructions="""
    You are an expert agronomist specializing in crop yield estimation, 
    crop rotation planning, and NDVI (Normalized Difference Vegetation Index) monitoring. 

        You specialize in crop yield prediction, crop rotation, and NDVI-based monitoring.
    
    - Always check if the user provides location (coordinates) and date range.
    - If they are missing, ask the user to provide them before making predictions.
    - If NDVI or other environmental data is available, analyze it and give insights on crop health, stress, and growth stage.
    - If data is insufficient, politely explain what additional info is required.
    - If the user asks about crop health, use NDVI ranges
      to provide meaningful recommendations.
    - If the user asks for advice on yield, irrigation, or fertilizer, 
      give context-aware guidance for better crop management.
    """,
    model_settings=ModelSettings(temperature=0.3)
)

economics_agent = base_agent.clone(
    name="AgriEconomist",
    instructions="Focus on agricultural economics, market trends, and profitability.",
    model_settings=ModelSettings(temperature=0.2)
)

async def route_question(question: str) -> List[Agent]:
    """Use LLM to decide which agents should answer."""
    router_prompt = f"""
    You are a router for agronomy questions.
    Decide which experts should answer the following question:
    "{question}"

    Available experts:
    - SoilExpert: soil health, fertility, pH, nutrients
    - CropExpert: crop yield, NDVI, crop monitoring, rotation
    - AgriEconomist: pricing, markets, profitability
    - Escalation: urgent, problem, disease, pest outbreak
    - BaseAgent: general agronomy knowledge

    Return a comma-separated list of experts.
    """

    # ✅ Correct way: use Runner.run with the base agent + router prompt
    router_result = await Runner.run(
        base_agent,
        router_prompt,
        max_turns=1
    )

    decision = router_result.final_output.lower()

    selected = []
    if "soil" in decision:
        selected.append(soil_agent)
    if "crop" in decision:
        selected.append(crop_agent)
    if "economist" in decision or "economics" in decision:
        selected.append(economics_agent)
    if "escalation" in decision or "urgent" in decision or "problem" in decision:
        return ["escalate"]  # special case
    if not selected:
        selected.append(base_agent)  # fallback

    return selected

async def run_agronomy_team(question: str) -> str:
    synthesis = ""
    agents_to_run = await route_question(question)

    if agents_to_run == ["escalate"]:
        result = await Runner.run(base_agent, question, max_turns=5)
        print(f"\n--- Escalation Triggered ---\n{result.final_output}\n")
        
        confirm = input("👉 Do you want to contact an agronomist? (yes/no): ").strip().lower()
        if confirm in ("yes", "y"):
            print("\n📞 Contact Agronomist:\nAleema Saleem\n+92 3378288720\n")
        else:
            print("\n👍 Okay, you can continue with the quick advice given above. Let me know if you need more guidance.\n")    

    else:
        collected = []
        for expert in agents_to_run:
            result = await Runner.run(expert, question, max_turns=5)
            collected.append(result.final_output)

        # Merge results into one response (remove duplicates & overlap)
        synthesis_prompt = f"""
        Combine the following expert answers into one clear, non-repetitive response
        for the user. Keep the most important actionable advice, and remove duplicates.

        Question: {question}

        Expert Answers:
        {' '.join(collected)}
        """

        synthesis_result = await Runner.run(base_agent, synthesis_prompt, max_turns=2)
        synthesis = synthesis_result.final_output
    return synthesis

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
