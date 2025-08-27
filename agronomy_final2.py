import os
import json
import asyncio
from dotenv import load_dotenv
import ee  # Google Earth Engine

from typing import List, Dict
from pydantic import BaseModel

# ===============
# Agents & Tools
# ===============
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

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.external_client = AsyncOpenAI(api_key=self.gemini_api_key, base_url=self.base_url)

        self.model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=self.external_client,
        )

ctx = Context()
ee.Initialize(project='crop-monitoring-project-469812')


# =============================
# Conversation Memory
# =============================
class ConversationMemory:
    def __init__(self, max_memory: int = 5):
        self.interaction_count = 0
        self.previous_queries = []
        self.max_memory = max_memory

    def add_query(self, query: str):
        self.interaction_count += 1
        self.previous_queries.append(query)
        if len(self.previous_queries) > self.max_memory:
            self.previous_queries = self.previous_queries[-self.max_memory:]

    def get_instructions(self, agent_name: str, role_desc: str) -> str:
        instr = f"You are {agent_name}, {role_desc}. Interaction #{self.interaction_count}.\n"
        if self.previous_queries:
            instr += "\nRecent queries (last 5 max):\n"
            for i, q in enumerate(self.previous_queries, 1):
                instr += f"{i}. {q}\n"
            instr += "\nRespond in continuity with earlier conversations.\n"
        return instr

memory = ConversationMemory(max_memory=5)


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


class RegionInput(BaseModel):
    coordinates: List[List[List[float]]]


@function_tool
async def crop_monitoring(region: RegionInput, start_date: str, end_date: str, index: str = "NDVI") -> str:
    try:
        polygon = ee.Geometry.Polygon(region.coordinates)
        collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
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


def is_user_admin(context: RunContextWrapper, agent) -> bool:
    user_role = getattr(context, "user_role", "user")
    return user_role == "admin"


@function_tool(is_enabled=is_user_admin)
def delete_field_data(none: str = "ok") -> str:
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
# Escalation Handling
# =============================
class EscalationData(BaseModel):
    reason: str
    field_id: str


async def on_escalation(ctx: RunContextWrapper, input_data: EscalationData):
    print(f"HANDOFF: Escalating field {input_data.field_id} due to {input_data.reason}")


escalation_agent = Agent(
    name="AgriEscalationAgent",
    instructions=lambda ctx, agent: memory.get_instructions(
        agent.name,
        "Handle urgent pest, disease, or critical agronomy problems with immediate solutions."
    )
)

escalation_handoff = handoff(
    agent=escalation_agent,
    tool_name_override="escalate_to_expert",
    tool_description_override="Escalate complex agronomy issues to a specialist.",
    on_handoff=on_escalation,
    input_type=EscalationData,
    input_filter=remove_all_tools
)


# =============================
# Base Agent with Tools
# =============================
base_agent = Agent(
    name="AgroDeepSearchAgent",
    instructions=lambda context, agent: memory.get_instructions(context, agent),
    tools=[agri_search, crop_monitoring, delete_field_data, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.4),
    tool_use_behavior=StopAtTools(stop_at_tool_names=["crop_monitoring"]),
    handoffs=[escalation_handoff]
)


# =============================
# Specialist Agents
# =============================
planner_agent = base_agent.clone(
    name="PlannerAgent",
    instructions="""
    You are a planning assistant.  
    - Break the user's query into specific sub-questions.
    - Return your output strictly as JSON: {"sub_questions": [ ... ]}
    """,
    model_settings=ModelSettings(temperature=0.3)
)

soil_agent = base_agent.clone(
    name="SoilExpert",
    instructions=lambda ctx, agent: memory.get_instructions(
        agent.name,
        """
        You are a soil health and fertility specialist. 
        Your job is to assess soil quality based on specific parameters and guide the user.

        ✅ Always ask the user to provide the following soil test data if missing:
        - Soil pH (acidity/alkalinity, affects nutrient solubility)
        - Electrical Conductivity (EC) → salinity level
        - Cation Exchange Capacity (CEC) → nutrient holding capacity
        - Organic Matter (OM) / Soil Organic Carbon (SOC)
        - Macronutrients: Nitrogen (N), Phosphorus (P), Potassium (K)
        - Secondary nutrients: Calcium (Ca), Magnesium (Mg), Sulfur (S)

        ✅ When values are provided:
        - Interpret whether each parameter is Low / Medium / High.
        - Explain its impact on nutrient availability or toxicity.
        - Recommend corrective actions (e.g., lime for low pH, gypsum for high sodium/EC, organic matter for low SOC).

        ❌ Never skip asking for missing parameters.
        ❌ Do not generate random values.
        ✅ Stay consistent and use standard agronomic guidelines.
        """
    ),
    model_settings=ModelSettings(temperature=0.25,)
)


# soil_agent = base_agent.clone(
#     name="SoilExpert",
#     instructions=lambda ctx, agent: memory.get_instructions(agent.name, "You specialize in soil health and fertility."),
#     model_settings=ModelSettings(temperature=0.2)
# )

geo_agent = base_agent.clone(
    name="GeoAgent",
    instructions=lambda ctx, agent: memory.get_instructions(agent.name, """
    You are a geospatial assistant that ONLY helps users provide location and time info.
    - If no coordinates, ask for latitude/longitude.
    - If no dates, ask for start/end dates.
    Return JSON with: {latitude, longitude, date_range}.
    ❌ Never calculate NDVI or analysis.
    """),
    model_settings=ModelSettings(temperature=0.2)
)

crop_agent = base_agent.clone(
    name="CropExpert",
    instructions=lambda ctx, agent: memory.get_instructions(agent.name, "You specialize in crop health, yield estimation, and NDVI monitoring."),
    model_settings=ModelSettings(temperature=0.3)
)

economics_agent = base_agent.clone(
    name="AgriEconomist",
    instructions=lambda ctx, agent: memory.get_instructions(agent.name, "Focus on agricultural economics and market trends."),
    model_settings=ModelSettings(temperature=0.2)
)


# =============================
# Routing Logic
# =============================
async def route_question(question: str) -> Dict[str, List[Agent]]:
    planner_result = await Runner.run(planner_agent, question, max_turns=2)
    try:
        plan = json.loads(planner_result.final_output.strip())
        sub_questions = plan.get("sub_questions", [question])
    except Exception:
        sub_questions = [question]

    assignments = {}
    for subq in sub_questions:
        router_prompt = f"""
        You are a router for agronomy questions.
        Decide which experts should answer: "{subq}".

        Available experts: SoilExpert, CropExpert, AgriEconomist, Escalation, BaseAgent
        Return a comma-separated list.
        """
        router_result = await Runner.run(base_agent, router_prompt, max_turns=1)
        decision = router_result.final_output.lower()

        selected = []
        if "soil" in decision: selected.append(soil_agent)
        if "crop" in decision: selected.append(crop_agent)
        if "economist" in decision or "economics" in decision: selected.append(economics_agent)
        if "escalation" in decision or "urgent" in decision: assignments[subq] = ["escalate"]; continue
        if not selected: selected.append(base_agent)

        assignments[subq] = selected
    return assignments


# =============================
# Team Orchestration
# =============================
async def run_agronomy_team(question: str) -> str:
    planner_result = await Runner.run(planner_agent, question, max_turns=2)
    try:
        plan = json.loads(planner_result.final_output.strip())
        sub_questions = plan.get("sub_questions", [question])
    except Exception:
        sub_questions = [question]

    async def process_subq(subq: str) -> str:
        local_syn = f"\n\n🔹 Sub-question: {subq}\n"
        assignments = await route_question(subq)
        experts = assignments.get(subq, [base_agent])

        if experts == ["escalate"]:
            result = await Runner.run(base_agent, subq, max_turns=5)
            return local_syn + f"\n--- Escalation Triggered ---\n{result.final_output}\n"

        async def run_expert(expert):
            if expert.name == "CropExpert":
                geo_result = await Runner.run(geo_agent, subq, max_turns=5)
                geo_output = geo_result.final_output.strip()
                out = f"\n--- GeoAgent ---\n{geo_output}\n"
                try:
                    geo_data = json.loads(geo_output)
                    if all(k in geo_data for k in ["latitude", "longitude", "date_range"]):
                        crop_prompt = f"User asked: {subq}\nGeoAgent data: {geo_output}\n"
                        crop_result = await Runner.run(crop_agent, crop_prompt, max_turns=5)
                        out += f"\n--- CropExpert ---\n{crop_result.final_output}\n"
                    else:
                        out += "\n❌ CropExpert skipped: incomplete geo data.\n"
                except json.JSONDecodeError:
                    crop_result = await Runner.run(crop_agent, subq, max_turns=5)
                    out += f"\n⚠️ GeoAgent unstructured, running CropExpert generally...\n"
                    out += f"\n--- CropExpert ---\n{crop_result.final_output}\n"
                return out
            else:
                result = await Runner.run(expert, subq, max_turns=5)
                return f"\n--- {expert.name} ---\n{result.final_output}\n"

        expert_outputs = await asyncio.gather(*(run_expert(ex) for ex in experts))
        return local_syn + "".join(expert_outputs)

    results = await asyncio.gather(*(process_subq(subq) for subq in sub_questions))
    return "".join(results)


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
                memory.add_query(question)
            print("\n💬 Thinking...\n")
            team_result = await run_agronomy_team(question)
            print(team_result)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break


if __name__ == "__main__":
    asyncio.run(main())
