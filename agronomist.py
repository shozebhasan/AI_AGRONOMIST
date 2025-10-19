import os
import json
import asyncio
from dotenv import load_dotenv
import ee  # Google Earth Engine
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from guards import check_for_abuse, handle_abuse_response

_context_cache = {}

from db import (
    init_db,
    async_save_message,
    async_get_history,
    async_get_or_create_user,
    async_get_messages_for_conversation,  
)

load_dotenv()  # load .env file
project_id = os.getenv("GEE_PROJECT_ID")
if project_id:
    try:
        ee.Initialize(project=project_id)
    except Exception as _:
        # Earth Engine init can fail in local dev without creds; do not crash
        pass

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


# Model and external clients

class Context:
    def __init__(self):
        load_dotenv()
        tracing_api_key = os.getenv("OPENAI_API_KEY")
        if tracing_api_key:
            set_tracing_export_api_key(tracing_api_key)

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        # Gemini-compatible OpenAI interface
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.external_client = AsyncOpenAI(api_key=self.gemini_api_key, base_url=self.base_url)

        self.model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=self.external_client,
        )

ctx = Context()


# Tools

@function_tool
async def agri_search(query: str) -> str:
    """Search for agronomy-related information"""
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
    """Monitor crop health using satellite imagery"""
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


# Memory building (global, DB-backed)

async def build_global_user_memory(user_id: int, max_messages: int = 200) -> str:
    """Build comprehensive user memory from all conversations"""
    history = await async_get_history(user_id, limit=max_messages)
    
    if not history or len(history) < 3:
        return "New user, no prior conversation history."
    
    # Build chronological conversation text
    lines = []
    for msg in reversed(history):
        role = "User" if msg.role == "user" else "Assistant"
        content = msg.content[:300]  # Limit to 300 chars per message
        lines.append(f"{role}: {content}")
    
    memory_source = "\n".join(lines)
    
    # Extract name heuristically
    import re
    name_match = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Za-z\s]{2,30})", memory_source, re.I)
    
    # Build structured extraction prompt
    memory_extraction_prompt = f"""You are extracting key user information from chat history.

IMPORTANT: Only extract facts the USER explicitly stated. Do not make assumptions.

Look for:
- User's name (if they said "my name is..." or "I am...")
- Location/region/country they mentioned
- Crops they grow or asked about (wheat, rice, tomatoes, etc.)
- Problems they've had (pests, diseases, soil issues)
- Farming preferences (organic, irrigation methods)

Conversation history:
{memory_source[:6000]}

Format your response EXACTLY like this example:
- Name: [name if mentioned, otherwise "Not provided"]
- Location: [location if mentioned, otherwise "Not provided"]  
- Crops: [list crops mentioned]
- Past Issues: [problems they discussed]
- Preferences: [any farming preferences mentioned]

Your extraction:"""
    
    temp_agent = Agent(
        name="MemoryExtractor",
        instructions="Extract user facts from conversation history. Be specific and accurate.",
        tools=[],
        model=ctx.model,
        model_settings=ModelSettings(temperature=0.1),
    )
    
    result = await Runner.run(temp_agent, memory_extraction_prompt, max_turns=1)
    memory_text = result.final_output.strip()
    
    
    if "not provided" in memory_text.lower() and len(memory_text) < 100:
        if name_match:
            memory_text = f"- Name: {name_match.group(1).strip()}\n" + memory_text
    
    # Persist to database
    try:
        from db import async_session_maker, User
        async with async_session_maker() as session:
            user_obj = await session.get(User, user_id)
            if user_obj:
                user_obj.memory = memory_text
                await session.commit()
                print(f"✅ Saved memory for user {user_id}: {memory_text[:100]}")
    except Exception as e:
        print(f"⚠️ Failed to persist memory: {e}")
    
    return memory_text





async def get_conversation_context(conversation_id: int, max_messages: int = 20) -> str:
    """Get recent conversation context with caching for performance"""
    if not conversation_id:
        return ""
    
    # Check cache (valid for 5 seconds)
    cache_key = f"{conversation_id}_{max_messages}"
    if cache_key in _context_cache:
        cached_data, timestamp = _context_cache[cache_key]
        if (datetime.utcnow() - timestamp).seconds < 5:
            return cached_data
    
    messages = await async_get_messages_for_conversation(conversation_id)
    if not messages:
        return ""
    
    # Limit to recent messages
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    
    context_lines = []
    for msg in recent:
        role = "You" if msg.role == "assistant" else "User"
        # Limit each message to 500 chars
        content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
        context_lines.append(f"{role}: {content}")
    
    result = "\n".join(context_lines)
    
    # Cache the result
    _context_cache[cache_key] = (result, datetime.utcnow())
    
    return result


# Escalation

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


# Agent factory

def create_base_agent(user_email: str, global_memory: str, conversation_context: str, vision_context: str = ""):
    """Create agent with full memory and conversation context"""
    
    instructions = f"""You are an expert AI Agronomist assistant helping {user_email.split('@')[0]}.

IMPORTANT - USER CONTEXT:
{global_memory}

RECENT CONVERSATION (refer to this for context):
{conversation_context}

{vision_context if vision_context else ""}

Your responsibilities:
- Reference past conversations naturally (e.g., "As we discussed before...")
- Remember user's crops, location, and farming methods
- Build upon previous advice given
- Use agri_search for research
- Use crop_monitoring for NDVI analysis
- Be conversational and remember what the user has told you

⚠️ LANGUAGE REQUIREMENT:
- Always understand Roman Urdu (Urdu written in Latin script, e.g. "ap kaise ho").
- If the user writes in Roman Urdu, respond back in Roman Urdu.
- If the user writes in English, respond in English.
- Never switch to Urdu script — stick to Roman Urdu when the user uses it.

When the user refers to "last time" or "before", check the conversation context above."""

    return Agent(
        name=f"AgroDeepSearchAgent_{user_email}",
        instructions=instructions,
        tools=[agri_search, crop_monitoring, counter_tool],
        model=ctx.model,
        model_settings=ModelSettings(temperature=0.4),
        tool_use_behavior=StopAtTools(stop_at_tool_names=["crop_monitoring"]),
        handoffs=[escalation_handoff]
    )

soil_agent = Agent(
    name="SoilExpert",
    instructions="You specialize in soil health and fertility. Provide actionable advice.",
    tools=[agri_search, crop_monitoring, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[escalation_handoff]
)

crop_agent = Agent(
    name="CropExpert",
    instructions="""
    You are an expert agronomist specializing in crop yield estimation,
    crop rotation planning, and NDVI monitoring.

    - Always check for location (coordinates) and date range.
    - If missing, ask the user before making predictions.
    - If NDVI or other data is available, analyze it and give insights on crop health, stress, and growth stage.
    - Explain what additional info is required if data is insufficient.
    - Give context-aware guidance for yield, irrigation, fertilizer.
    """,
    tools=[agri_search, crop_monitoring, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.3),
    handoffs=[escalation_handoff]
)

economics_agent = Agent(
    name="AgriEconomist",
    instructions="Focus on agricultural economics, market trends, and profitability.",
    tools=[agri_search, crop_monitoring, counter_tool],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[escalation_handoff]
)


# Router

async def route_question_with_context(question: str, base_agent: Agent, context_agents: dict) -> List[Agent]:
    """Route question to appropriate experts with memory context"""
    
    # Check for urgent/escalation keywords first
    urgent_keywords = ["urgent", "emergency", "dying", "rotting", "serious condition", "help immediately", "crisis", "problem"]
    if any(keyword in question.lower() for keyword in urgent_keywords):
        print(f"🚨 DEBUG: Detected urgent issue - routing to escalation")
        return ["escalate"]
    
    router_prompt = f"""Which experts should answer: "{question}"
    Options: SoilExpert, CropExpert, AgriEconomist, BaseAgent
    Return comma-separated list."""
    
    router_result = await Runner.run(base_agent, router_prompt, max_turns=1)
    decision = router_result.final_output.lower()

    selected: List[Agent] = []
    if "soil" in decision:
        selected.append(context_agents['soil'])
    if "crop" in decision:
        selected.append(context_agents['crop'])
    if "economist" in decision or "economics" in decision:
        selected.append(context_agents['economics'])
    
    if not selected:
        selected.append(base_agent)
    
    print(f"🔍 DEBUG: Routing to: {[a.name for a in selected]}")
    return selected


# Orchestrator with memory

async def run_agronomy_team(
    question: str,
    user_email: str,
    user_id: int,
    conversation_id: Optional[int] = None,
    language: Optional[str] = "en",
    images: Optional[List[str]] = None,  
    vision_data: Optional[Dict[str, Any]] = None
) -> str:
    """Run agronomy team with full memory context - optimized for speed"""
    
    # Load persisted memory (fast - just database read)
    try:
        from db import async_session_maker, User
        async with async_session_maker() as session:
            user_obj = await session.get(User, user_id)
            persisted_memory = getattr(user_obj, "memory", None) if user_obj else None

            
            print(f"📝 DEBUG: Loaded memory from DB: {persisted_memory[:100] if persisted_memory else 'NONE'}")
    except Exception as e:
        print(f"⚠️ Could not read persisted memory: {e}")
        persisted_memory = None
    
    # Use cached memory or trigger async build
    if not persisted_memory or len(persisted_memory) < 50:
        # Build in background, don't block the response
        asyncio.create_task(build_global_user_memory(user_id, max_messages=200))
        global_memory = persisted_memory or ""  # Use what we have, even if empty
    else:
        global_memory = persisted_memory
        # Refresh memory periodically in background
        recent_history = await async_get_history(user_id, limit=10)
        if len(recent_history) >= 10:
            asyncio.create_task(build_global_user_memory(user_id, max_messages=200))

    print(f"🧠 DEBUG: Using global memory: {global_memory[:200]}")
    
    # Get conversation-specific context (with caching)
    conversation_context = ""
    if conversation_id:
        conversation_context = await get_conversation_context(conversation_id, max_messages=15)
        print(f"🔍 DEBUG: Loaded conversation context ({len(conversation_context)} chars)")
        print(f"🔍 DEBUG: Global memory preview: {global_memory[:300] if global_memory else 'EMPTY'}")

    # Build vision context string if available
    vision_context = ""
    if vision_data:
        vision_context = f"""
🔬 VISION ANALYSIS RESULTS:
- Detected Crop: {vision_data.get('crop', 'Unknown')}
- Health Status: {vision_data.get('status', 'Unknown')}
- Condition/Disease: {vision_data.get('label', 'Unknown')}
- Detection Confidence: {vision_data.get('confidence', 0):.1%}
- Analysis Mode: {vision_data.get('mode', 'unknown')}
"""
        if vision_data.get('advice'):
            vision_context += f"- Recommended Action: {vision_data['advice']}\n"
        
        if vision_data.get('mode') == 'auto_uncertain':
            vision_context += "\n⚠️ NOTE: Auto-detection confidence was low. Results are best-guess predictions.\n"
        
        print(f"🔬 Vision context added: {vision_data.get('crop')} - {vision_data.get('label')}")
    
    # Create personalized base agent
    base_agent = create_base_agent(user_email, global_memory, conversation_context, vision_context)
    print(f"🔍 DEBUG: Agent instructions length: {len(base_agent.instructions) if isinstance(base_agent.instructions, str) else 'DYNAMIC'}")
    
    # Prepare question with image context
    if images and len(images) > 0:
        image_context = f"\n\n[User has shared {len(images)} image(s) for analysis. Please analyze the visual information and provide relevant agricultural advice.]\n"
        question_for_model = image_context + question
    else:
        question_for_model = question
    # Prepare the core question for the agent
    if language and language.startswith("ur"):
        
        
        question_for_model = (
            f"{question}\n\nIMPORTANT: Reply in Urdu using standard Arabic script. "
            "Do NOT use Roman Urdu or transliteration. Provide concise, natural Urdu text only. "
            "Provide a complete, detailed, and natural explanation in Urdu,"
            "matching the depth and richness you would normally give in English. "
            "Use clear, professional Urdu suitable for farmers and agronomists."
        )
    else:
        question_for_model = question
    
    # Build message
    user_message = {
        "role": "user",
        "content": question_for_model
    }

    # If images provided, add them to the message (Gemini supports multimodal)
    if images and len(images) > 0:
        # For Gemini, we can include images in the content
        # Format: content can be array of text and image parts
        user_message["content"] = [
            {"type": "text", "text": question_for_model}
        ]
        for img_base64 in images[:4]:  # Limit to 4 images
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })

    messages = [user_message]
    
    # Fast escalation check (keyword-based, no routing needed)
    urgent_keywords = ["urgent", "emergency", "dying", "rotting", "serious condition", "crisis", "problem", "help immediately"]
    if any(kw in question.lower() for kw in urgent_keywords):
        print(f"🚨 Escalation detected")

        # Use simple string prompt instead of messages list
        escalation_prompt = f"""You are helping with an urgent agricultural issue.
        
    User Context: {global_memory}
    Recent Conversation: {conversation_context}

    User's urgent question: {question}
    {f"[User has provided {len(images)} image(s) showing the issue]" if images else ""}
    Provide immediate, practical advice to address this urgent situation. Be specific and actionable."""
        
        try:
            result = await Runner.run(base_agent, escalation_prompt, max_turns=3)
            response = result.final_output

        except Exception as e:
             print(f"❌ Escalation error: {e}")
             response = "I understand this is urgent. Here's immediate advice:\n\n1. Assess the situation quickly\n2. Take photos if possible\n3. Check for visible pests or diseases\n4. Ensure proper irrigation"

        return f"{response}\n\n📞 Professional Agronomist Contact:\n• Name: Aleema Saleem\n• Phone: +92 3378288720\n• WhatsApp: wa.me/923378288720\n• Availability: For urgent agricultural assistance"
    # Simplified routing - one LLM call
    router_prompt = f"""Analyze this question and respond with ONE word only: Soil, Crop, Economics, or General.

Question: "{question}"

Your response (one word only):"""
    
    router_result = await Runner.run(base_agent, router_prompt, max_turns=1)
    decision = router_result.final_output.strip().lower()
    print(f"🔍 DEBUG: Routing decision: {decision}")
    
    # Create only the needed specialized agent (lazy loading)
    if "soil" in decision:
        expert = Agent(
            name="SoilExpert",
            instructions=f"""You are a soil health specialist with image analysis capabilities.

USER CONTEXT:
{global_memory}

RECENT CONVERSATION:
{conversation_context}
{f"The user has shared {len(images)} image(s). Analyze any visible soil conditions, texture, color, or issues in the images." if images else ""}

Provide actionable soil advice while remembering the user's context.""",
            tools=[agri_search, crop_monitoring, counter_tool],
            model=ctx.model,
            model_settings=ModelSettings(temperature=0.2),
            handoffs=[escalation_handoff]
        )
    elif "crop" in decision:
        expert = Agent(
            name="CropExpert",
            instructions=f"""You are a crop yield and monitoring specialist with plant disease identification capabilities.

USER CONTEXT:
{global_memory}

RECENT CONVERSATION:
{conversation_context}

{vision_context if vision_context else ""}

{f"The user has shared {len(images)} image(s). Carefully analyze the plants, leaves, stems, or symptoms visible in the images. Identify any diseases, pests, nutrient deficiencies, or growth issues." if images else ""}

Analyze crop health and provide growth insights while remembering context.""",
            tools=[agri_search, crop_monitoring, counter_tool],
            model=ctx.model,
            model_settings=ModelSettings(temperature=0.3),
            handoffs=[escalation_handoff]
        )
    elif "economics" in decision or "economic" in decision:
        expert = Agent(
            name="AgriEconomist",
            instructions=f"""You are an agricultural economics specialist.

USER CONTEXT:
{global_memory}

RECENT CONVERSATION:
{conversation_context}

Focus on market trends and profitability while remembering user context.""",
            tools=[agri_search, crop_monitoring, counter_tool],
            model=ctx.model,
            model_settings=ModelSettings(temperature=0.2),
            handoffs=[escalation_handoff]
        )
    else:
        # Use base agent for general questions
        expert = base_agent
    
    # Run the selected expert
    result = await Runner.run(expert, messages, max_turns=5)
    return result.final_output


#new code
# ---------- translation helper (uses existing agent runner) ----------
# simple in-memory cache to reduce duplicate translations
_TRANSLATION_CACHE: Dict[str, str] = {}

async def translate_to_urdu_batch(texts: List[str]) -> List[str]:
    """
    Translate a list of short texts to Urdu (Arabic script) in a single call.
    We batch them into one prompt and parse a JSON array response to save API calls.
    Returns list of translations in same order; on failure returns originals.
    """
    if not texts:
        return []

    # prepare list of texts that are not cached
    to_translate = []
    indexes = []
    results: List[str] = [None] * len(texts)  # type: ignore

    for i, t in enumerate(texts):
        if not t:
            results[i] = t
        elif t in _TRANSLATION_CACHE:
            results[i] = _TRANSLATION_CACHE[t]
        else:
            indexes.append(i)
            to_translate.append(t)

    if not to_translate:
        # all were cached or empty
        return results  # type: ignore

    # Build a single prompt listing all texts and asking for JSON output
    # We enumerate to_translate to keep a compact prompt; model should return a JSON array
    prompt_items = "\n".join(f"{idx}: {text.replace(chr(10),' ')}" for idx, text in enumerate(to_translate))
    prompt = (
        "Translate the following items to Urdu using Arabic script. "
        "Return a JSON array of translations in the same order, and return only valid JSON.\n\n"
        f"Items:\n{prompt_items}\n\nOutput JSON:"
    )

    try:
        # Create a lightweight temporary translator agent and run it.
        # NOTE: this assumes Agent, ModelSettings, ctx and Runner are available in this module.
        try:
            translator_agent = Agent(
                name="TranslatorAgent",
                instructions=(
                    "You are a concise translator. Translate the provided items into Urdu using standard Arabic script. "
                    "Return only a JSON array of translated strings in the same order as the input. Do not add any commentary."
                ),
                tools=[],
                model=ctx.model,
                model_settings=ModelSettings(temperature=0.0),
                handoffs=[],
            )

            res = await Runner.run(translator_agent, [{"role": "user", "content": prompt}], max_turns=1)

        except Exception as create_err:
            # If creating the translator agent fails, try to fall back to any available agent in globals
            # or fall back to a direct Runner.run attempt.
            print("translate_to_urdu_batch: translator agent creation failed, attempting fallback...", create_err)
            agent_var = None
            for candidate in ("base_agent", "expert", "agronomist_agent", "agent", "expert_agent"):
                if candidate in globals():
                    agent_var = globals()[candidate]
                    break

            if agent_var is not None:
                res = await Runner.run(agent_var, [{"role": "user", "content": prompt}], max_turns=1)
            else:
                # Last-resort attempt: if Runner supports being called with messages only (some runner APIs do),
                # try that. If it fails, raise a clear error so logs show what to fix.
                try:
                    res = await Runner.run([{"role": "user", "content": prompt}], max_turns=1)
                except Exception as fallback_err:
                    raise RuntimeError(
                        "translate_to_urdu_batch error: no available agent for translation. "
                        "Ensure Agent, ModelSettings, ctx.model and Runner are defined in this module."
                    ) from fallback_err

        raw = (getattr(res, "final_output", None) or "").strip()

        # try to extract JSON from response
        translations: List[str] = []
        try:
            translations = json.loads(raw)
            if not isinstance(translations, list):
                raise ValueError("Expected JSON array")
        except Exception:
            # fallback: split lines — not ideal, but safe
            translations = [line.strip() for line in raw.splitlines() if line.strip()]

        # fill results and cache
        for idx, text_index in enumerate(indexes):
            translated = translations[idx] if idx < len(translations) else to_translate[idx]
            # ensure we have a string
            if translated is None:
                translated = to_translate[idx]
            _TRANSLATION_CACHE[to_translate[idx]] = translated
            results[text_index] = translated

        # for any remaining None, keep original
        for i, r in enumerate(results):
            if r is None:
                results[i] = texts[i]

        return results  # type: ignore

    except Exception as e:
        print("translate_to_urdu_batch error:", e)
        # fallback: return originals
        return texts




# API-facing processing

async def process_with_agronomy_team(message: str, email: str, conversation_id: Optional[int] = None, language: Optional[str] = "en", images: Optional[List[str]] = None,vision_data: Optional[Dict[str, Any]] = None ) -> str:
    """Process messages with full memory context and image support
    language: 'en' or 'ur' (or other codes) — 'ur' will instruct the agent to reply in Urdu (Arabic script).
    """
    try:
        #Guardrail
        if check_for_abuse(message):
            return handle_abuse_response()
        
        user = await async_get_or_create_user(email, email.split('@')[0])
        
        if (
            vision_data
            and vision_data.get("mode") == "auto_uncertain"
            and images
            and len(images) > 0
        ):
            msg_lower = message.lower()
            confirmed_crop = None
            for crop in ("wheat", "corn", "rice", "cotton"):
                if crop in msg_lower:
                    confirmed_crop = crop
                    break
            if confirmed_crop:
                print(f"[agronomy] User confirmed crop: {confirmed_crop}")
                from vision import predict_crop_from_base64, _get_advice_for_label

                result = predict_crop_from_base64(images[0], confirmed_crop)
                advice = _get_advice_for_label(confirmed_crop, result["label"])
                result["advice"] = advice
                result["mode"] = "manual"
                result["crop_confidence"] = vision_data.get("crop_confidence", 0.0)
                vision_data = result  # ✅ Replace with confirmed analysis
        response = await run_agronomy_team(
            question=message,
            user_email=email,
            user_id=user.id,
            conversation_id=conversation_id,
            language=language,
            images=images,
            vision_data=vision_data
        )
        return response
    except Exception as e:
        print(f"❌ Error in process_with_agronomy_team: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."

async def main() -> None:
    print("🌱 Agronomy Deep Research Agent ready! Type 'exit' to quit.\n")
    await init_db()
    user_email = input("👤 Enter your email address: ").strip() or "guest@example.com"

    user = await async_get_or_create_user(user_email, user_email.split('@')[0])
    print(f"✅ Welcome, {user.name}! Your chats will be saved under user_id={user.id}\n")

    # Use database for conversation tracking
    from db import async_create_conversation
    conversation_id = await async_create_conversation(user.id, "Terminal Session")
   

    while True:
        try:
            question = input("❓ Ask your agronomy question: ").strip()
            if question.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            if question.startswith("/history"):
                try:
                    limit = int(question.split(" ")[1])
                except (IndexError, ValueError):
                    limit = 10
                history = await async_get_history(user.id, limit)
                print("\n📜 Your recent chat history:")
                for h in history:
                    role = "👤 You" if h.role == 'user' else "🤖 Agronomist"
                    print(f"{role}: {h.content}")
                print("")
                continue

            if question.startswith("/new"):
                conversation_id = await async_create_conversation(user.id, "Terminal Session")
                print("🆕 New conversation started!")
                continue

            print("\n💬 Thinking...\n")
            team_result = await run_agronomy_team(question, user_email, user.id, conversation_id)

            # Save to database
            await async_save_message(user.id, "user", question, conversation_id)
            await async_save_message(user.id, "assistant", team_result, conversation_id)

            print(team_result)
            print("\n" + "="*50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break

if __name__ == "__main__":
    asyncio.run(main())
