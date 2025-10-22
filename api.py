import hashlib
import asyncio
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import base64
import io
import traceback
from fastapi import File, UploadFile, Form
from PIL import Image
from vision import analyze_image_auto

from fastapi import FastAPI
from fastapi import Path
from fastapi import Request
from fastapi import Query     #del
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from agronomist import process_with_agronomy_team
from db import (
    init_db,
    check_db_health,
    async_get_or_create_user,
    async_create_conversation,
    async_save_message,
    async_get_history,
    async_get_conversations_for_user,
    async_get_messages_for_conversation,
    async_session_maker,
    User,
    Conversation,
    async_delete_conversation,
    async_create_password_reset_token,
    async_update_user_password,
    async_verify_password_reset_token,
    async_use_password_reset_token,
    get_user_facts,
    get_recent_memory,
    add_memory_entry,
    save_user_fact
    
)

load_dotenv()

app = FastAPI(
    title="Agronomist AI API",
    description="Backend API for the Agronomist AI application",
    version="1.0.0",
)


# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models

class ChatRequest(BaseModel):
    message: str
    email: str
    name: Optional[str] = None
    conversation_id: Optional[int] = None
    language: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded images

    class Config:
        extra = "allow"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    success: bool = True
    error: Optional[str] = None

class HistoryResponse(BaseModel):
    success: bool
    history: List[dict]
    conversations: List[dict]
    error: Optional[str] = None

class NewChatRequest(BaseModel):
    email: str
    name: Optional[str] = None

class NewChatResponse(BaseModel):
    success: bool
    conversation_id: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str

# Add to api.py after existing endpoints

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyResetCodeRequest(BaseModel):
    email: str
    code: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Send password reset code to user's email"""
    try:
        # Check if user exists
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalars().first()
            
            if not user:
                # Don't reveal whether email exists or not for security
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "message": "If the email exists, a reset code has been sent."}
                )
        
        # Generate and save reset token
        reset_code = await async_create_password_reset_token(request.email)
        
        # Send email
        from email_service import email_service
        email_sent = await email_service.send_password_reset_email(request.email, reset_code)
        
        if email_sent:
            return {
                "success": True, 
                "message": "Password reset code has been sent to your email."
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False, 
                    "error": "Failed to send email. Please try again later."
                }
            )
        
    except Exception as e:
        print(f"❌ Forgot password error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to process reset request"}
        )

@app.post("/api/auth/verify-reset-code")
async def verify_reset_code(request: VerifyResetCodeRequest):
    """Verify the reset code"""
    try:
        is_valid = await async_verify_password_reset_token(request.email, request.code)
        
        if is_valid:
            return {"success": True, "message": "Code verified successfully"}
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid or expired reset code"}
            )
            
    except Exception as e:
        print(f"❌ Verify reset code error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to verify code"}
        )

@app.post("/api/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset user's password"""
    try:
        # Verify code first
        is_valid = await async_verify_password_reset_token(request.email, request.code)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid or expired reset code"}
            )
        
        # Validate password length
        if len(request.new_password) < 6:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Password must be at least 6 characters"}
            )
        
        # Update password
        success = await async_update_user_password(request.email, request.new_password)
        if not success:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "User not found"}
            )
        
        # Mark token as used
        await async_use_password_reset_token(request.email, request.code)
        
        return {"success": True, "message": "Password reset successfully"}
        
    except Exception as e:
        print(f"❌ Reset password error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to reset password"}
        )


@app.post("/api/memory/refresh")
async def refresh_user_memory(request:Request):
    """Background endpoint to refresh user memory"""
    try:
        body = await request.json()
        email = body.get("email")
        if not email:
            return {"success": False, "error": "Email required"}
        user = await async_get_or_create_user(email, email.split("@")[0])

        # Import here to avoid circular dependency
        from agronomist import build_global_user_memory

        # Build memory in background (don't await)
        asyncio.create_task(build_global_user_memory(user.id, max_messages=200))

        return {"success": True, "message": "Memory refresh started"}
    except Exception as e:
        print(f"Memory refresh error: {e}")
        return {"success": False, "error": str(e)}

# Auth models
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    user: Optional[dict] = None
    error: Optional[str] = None
    message: Optional[str] = None


# Password helpers

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    return hash_password(plain_password) == hashed_password


# Startup

@app.on_event("startup")
async def startup_event():
    try:
        await init_db()
        healthy = await check_db_health()
        if healthy:
            print("✅ Database initialized successfully")
        else:
            print("⚠️ Database connection issues")
    except Exception as e:
        print(f"⚠️ DB init warning: {e}")


# Authentication Endpoints

@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    try:
        if not request.email or not request.password:
            return AuthResponse(success=False, error="Email and password required")

        if len(request.password) < 6:
            return AuthResponse(success=False, error="Password must be at least 6 characters")

        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            existing = result.scalars().first()
            if existing:
                return AuthResponse(success=False, error="Email already registered")

            new_user = User(email=request.email, name=request.email.split("@")[0])
            setattr(new_user, "hashed_password", hash_password(request.password))
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)

            return AuthResponse(
                success=True,
                user={"email": new_user.email, "name": new_user.name},
                message="Account created successfully",
            )
    except Exception as e:
        print(f"❌ Signup error: {e}")
        return AuthResponse(success=False, error="Registration failed")
    

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    try:
        if not request.email or not request.password:
            return AuthResponse(success=False, error="Email and password required")

        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalars().first()

            if not user:
                return AuthResponse(success=False, error="User not found")

            if not verify_password(request.password, getattr(user, "hashed_password", "")):
                return AuthResponse(success=False, error="Invalid credentials")

            return AuthResponse(success=True, user={"email": user.email, "name": user.name}, message="Login successful")
    except Exception as e:
        print(f"❌ Login error: {e}")
        return AuthResponse(success=False, error="Authentication failed")

@app.get("/api/auth/me", response_model=AuthResponse)
async def get_me(email: str):
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalars().first()
            if not user:
                return AuthResponse(success=False, error="User not found")
            return AuthResponse(success=True, user={"email": user.email, "name": user.name})
    except Exception as e:
        print(f"❌ /api/auth/me error: {e}")
        return AuthResponse(success=False, error=str(e))


# Health

@app.get("/", response_model=HealthResponse)
async def root():
    db_status = "healthy" if await check_db_health() else "degraded"
    return {"status": "healthy", "service": "Agronomist AI API", "version": "1.0.0", "database": db_status}

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    db_status = "healthy" if await check_db_health() else "degraded"
    return {"status": "healthy", "service": "Agronomist AI API", "version": "1.0.0", "database": db_status}


# Chat Endpoints

@app.post("/api/chat/new", response_model=NewChatResponse)
async def start_new_chat(request: NewChatRequest):
    try:
        user = await async_get_or_create_user(request.email, request.name or request.email.split("@")[0])
        conv_id = await async_create_conversation(user.id, "New Chat")
        
        return NewChatResponse(success=True, conversation_id=str(conv_id))
    except Exception as e:
        print(f"❌ New chat error: {e}")
        return NewChatResponse(success=False, conversation_id="", error=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: Request):
    try:
        body_json = await request.json()
        
        # Convert conversation_id to int 
        if "conversation_id" in body_json and body_json["conversation_id"] is not None:
            try:
                if isinstance(body_json["conversation_id"], str):
                    body_json["conversation_id"] = int(body_json["conversation_id"])
            except (ValueError, TypeError):
                body_json["conversation_id"] = None

        chat_req = ChatRequest(**(body_json or {}))
        language = (chat_req.language or "en").lower()
        user = await async_get_or_create_user(chat_req.email, chat_req.name or chat_req.email.split("@")[0])

        # Handle conversation ID
        if chat_req.conversation_id:
            db_conversation_id = chat_req.conversation_id
        else:
            db_conversation_id = await async_create_conversation(user.id, "New Chat")

        # Process images if present
        detection_summary = None
        vision_results = None  
        
        if chat_req.images and len(chat_req.images) > 0:
            try:
                # Always use auto pipeline: crop classifier → disease model
                start_vision = time.perf_counter()
                analysis = analyze_image_auto(chat_req.images[0], require_threshold=0.60)
                end_vision = time.perf_counter()
                print(f"⏱️ Vision analysis took {end_vision - start_vision:.2f} seconds")

                if analysis["mode"] == "rejected":
                    vision_results = analysis
                    detection_summary = (
                        f"⚠️ Unable to analyze this image.\n"
                        f"{analysis['description']}\n\n"
                        f"{analysis['advice']}"

                    )

                elif analysis["mode"] == "auto_uncertain" :
                    vision_results = analysis
                    detection_summary = (
                        f"⚠️ The crop classifier was not confident enough to proceed.\n"
                        f"{analysis['description']}\n\n"
                        f"{analysis['advice']}"
                    )

                elif analysis["mode"] == "auto_uncertain_dual":
                    vision_results = analysis
                    detection_summary = (
                        f"⚠️ The crop classifier was uncertain between two crops.\n\n"
                        f"**Prediction 1:** {analysis['crop_1'].capitalize()} — {analysis['label_1']} ({analysis['status_1']})\n"
                        f"{analysis['advice_1']}\n\n"
                        f"**Prediction 2:** {analysis['crop_2'].capitalize()} — {analysis['label_2']} ({analysis['status_2']})\n"
                        f"{analysis['advice_2']}"
                    )
                
                else:
                    status = "Healthy" if "healthy" in str(analysis['label']).lower() else "Infected"
                    vision_results = {
                        "mode": analysis.get("mode", "auto"),
                        "crop": analysis['crop'],
                        "status": status,
                        "label": analysis['label'],
                        "confidence": analysis['confidence'],
                        "description": analysis.get('description', ''),
                        "advice": analysis.get('advice', ''),
                        "uncertain": analysis.get("uncertain", False)
                    }

                    if vision_results.get("uncertain", False):
                        detection_summary = (
                            f"⚠️ The disease model was not confident in its prediction.\n"
                            f"It suggested **{vision_results['label']}**, but the confidence was only {vision_results['confidence']:.2f}.\n"
                            f"This may not be accurate. Please confirm the crop type or upload a clearer image."
                        )
                    else:

                        detection_summary = (
                            f"Crop: {vision_results['crop']}\n"
                            f"Disease: {vision_results['label']}\n\n"
                            f"{vision_results['advice']}"

                        )
            except Exception as vis_e:
                print(f"❌ Vision detection error: {vis_e}")
                import traceback
                traceback.print_exc()
                detection_summary = f"[Vision error: {str(vis_e)}]"
                vision_results = None
            

        # message for agent
        full_message = chat_req.message or "Analyze this crop image."
        
        # Adding vision context to message 
        if detection_summary:
            full_message = f"[VISION ANALYSIS]\n{detection_summary}\n\n[USER MESSAGE]\n{full_message}"

        # Save user message 
        user_message_to_save = chat_req.message or "[Image uploaded]"
        if chat_req.images:
            user_message_to_save = f"[Image] {user_message_to_save}"
        await async_save_message(user.id, "user", user_message_to_save, db_conversation_id)

        # Inject recent assistant memory only
        recent_memory = await get_recent_memory(user.id)
        if recent_memory:
            full_message = f"[RECENT MEMORY]\n{recent_memory}\n\n[USER MESSAGE]\n{full_message}"

        # Extract and store persistent user facts
        msg_lower = (chat_req.message or "").lower()

        # Extract name
        if "my name is" in msg_lower:
            try:
                name_part = msg_lower.split("my name is")[-1].strip()
                name = name_part.split()[0].capitalize()
                await save_user_fact(user.id, "name", name)
            except Exception as e:
                print(f"❌ Name extraction failed: {e}")

        # Extract crop interests
        crop_phrases = ["i grow", "i farm", "my crops are", "i cultivate"]
        for phrase in crop_phrases:
            if phrase in msg_lower:
                try:
                    crop_part = msg_lower.split(phrase)[-1].split(".")[0].strip()
                    await save_user_fact(user.id, "crop_interests", crop_part)
                    break
                except Exception as e:
                    print(f"❌ Crop extraction failed: {e}")

        # Conditional recall of persistent facts
        if "what's my name" in msg_lower or "do you know my name" in msg_lower:
            facts = await get_user_facts(user.id)
            name = facts.get("name", None)
            response_text = f"Your name is {name}." if name else "I don't have your name saved yet."
            await async_save_message(user.id, "assistant", response_text, db_conversation_id)
            return ChatResponse(response=response_text, conversation_id=str(db_conversation_id), success=True)

        if "what crops do i grow" in msg_lower or "what are my crops" in msg_lower or "do you remember my crops" in msg_lower:
            facts = await get_user_facts(user.id)
            crops = facts.get("crop_interests", None)
            response_text = f"You grow {crops}." if crops else "I don't have your crop interests saved yet."
            await async_save_message(user.id, "assistant", response_text, db_conversation_id)
            return ChatResponse(response=response_text, conversation_id=str(db_conversation_id), success=True)



        # Process with agronomy team pass vision_results if available
        start_agent = time.perf_counter()
        response_text = await process_with_agronomy_team(
            message=full_message,
            email=chat_req.email,
            conversation_id=db_conversation_id,
            language=language,
            vision_data=vision_results  
        )
        end_agent = time.perf_counter()
        print(f"⏱️ Agent response took {end_agent - start_agent:.2f} seconds")

        # Save assistant response
        await async_save_message(user.id, "assistant", response_text, db_conversation_id)

        await add_memory_entry(user.id, response_text)



        # Update conversation title 
        async with async_session_maker() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == db_conversation_id)
            )
            conv = result.scalars().first()
            if conv and conv.title == "New Chat":
                # choses between crop1 and crop 2 
                if vision_results:
                    if vision_results.get("mode") == "auto_uncertain_dual":
                        new_title = f"{vision_results['crop_1']} or {vision_results['crop_2']}"

                    elif vision_results.get("mode") == "rejected" :
                        new_title = "Uncertain Crop"

                    elif vision_results.get("mode") == "auto_certain" :
                        new_title = f"Uncertain - {vision_results['crop']}"
                    else:
                        new_title = f"{vision_results['crop']} - {vision_results['label']}"
                    
                    
                else:
                    # Fallback to message-based title
                    msg = chat_req.message or "Image Analysis"
                    new_title = msg[:50] + ("..." if len(msg) > 50 else "")
                conv.title = new_title
                await session.commit()

        return ChatResponse(
            response=response_text, 
            conversation_id=str(db_conversation_id), 
            success=True,
            metadata={"vision_results": vision_results} if vision_results else None
        )

    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response="Error processing request", 
            conversation_id="", 
            success=False, 
            error=str(e)
        )


    
#delete conversations in sidebar

@app.delete("/api/conversation/{user_email}/{conversation_id}")
async def delete_conversation(user_email: str, conversation_id: str = Path(...)):
    try:
        # validate user exists (optional)
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid conversation ID"})

        deleted = await async_delete_conversation(conv_id_int)
        if not deleted:
            return JSONResponse(status_code=404, content={"success": False, "error": "Conversation not found"})

        # Return remaining conversations so frontend can refresh easily (optional)
        conversations = await async_get_conversations_for_user(user.id)
        return JSONResponse(status_code=200, content={"success": True, "conversations": conversations})
    except Exception as e:
        print(f"❌ delete_conversation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})    

# ---------------------------
# History & Conversation Endpoints (Fixed)
# ---------------------------
@app.get("/api/history/{user_email}", response_model=HistoryResponse)
async def get_chat_history(user_email: str, limit: int = 20):  # Reduced limit
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        # Get conversations (fast query with index)
        conversations = await async_get_conversations_for_user(user.id)
        conversations.sort(
            key=lambda x: x.get("last_message_at") or x.get("created_at") or "",
            reverse=True,
        )

        # Only get recent messages (reduced from 50 to 20)
        db_messages = await async_get_history(user.id, limit=limit)
        history_data = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "conversation_id": msg.conversation_id,
            }
            for msg in db_messages
        ]

        return HistoryResponse(
            success=True,
            conversations=conversations,
            history=history_data,
        )

    except Exception as e:
        print(f"❌ History error: {e}")
        return HistoryResponse(
            success=False,
            conversations=[],
            history=[],
            error=str(e),
        )




@app.get("/api/conversation/{user_email}/{conversation_id}")
async def get_conversation(user_email: str, conversation_id: str):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid conversation ID"})

        # Get messages from DB
        messages = await async_get_messages_for_conversation(conv_id_int)
        if not messages:
            return JSONResponse(status_code=404, content={"success": False, "error": "No messages found for this conversation"})

        msgs_out = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "conversation_id": m.conversation_id,
            }
            for m in messages
        ]

        # Get conversation metadata
        async with async_session_maker() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conv_id_int)
            )
            conv = result.scalars().first()
            if conv:
                conv_meta = {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None
                }
            else:
                return JSONResponse(status_code=404, content={"success": False, "error": "Conversation not found"})

        return JSONResponse(status_code=200, content={"success": True, "conversation": conv_meta, "messages": msgs_out})
    except Exception as e:
        print(f"❌ get_conversation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})



# Session Utilities

@app.post("/api/user")
async def create_or_get_user(request: ChatRequest):
    try:
        user = await async_get_or_create_user(request.email, request.name or request.email.split("@")[0])
        return {"success": True, "user": {"id": user.id, "email": user.email, "name": user.name}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Main runner

def main():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

if __name__ == "__main__":
    main()