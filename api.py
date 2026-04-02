import hashlib
import time
from typing import List, Optional
import base64
from vision import analyze_image_auto
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO
from fastapi import File, UploadFile, Form
import tempfile
import os
os.environ["PATH"] = r"C:\Users\AFZAL COMPUTERS\AppData\Local\Microsoft\WinGet\Links" + os.pathsep + os.environ["PATH"]
import whisper
from fastapi import FastAPI
from fastapi import Path
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from rag import ingest_pdf, get_chunk_count
import shutil
from fastapi.responses import StreamingResponse

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
    save_user_fact,
    save_chat_image,
    async_delete_account,
    async_save_message_feedback
    
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

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyResetCodeRequest(BaseModel):
    email: str
    code: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

class MessageFeedbackRequest(BaseModel):
    message_id: int
    feedback_type: str  # 'good' or 'bad'
    comment: Optional[str] = None

class DeleteAccountRequest(BaseModel):
    email: str
    password: str  # require password confirmation for safety

class TranscribeRequest(BaseModel):
    audio: str  # Base64 encoded audio
    filename: Optional[str] = "audio.webm"

class TranscribeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None

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

WHISPER_MODEL = None

#whisper model
def load_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("📥 Loading Whisper model (this may take a moment on first run)...")
        WHISPER_MODEL = whisper.load_model("base")  # Change to "small" or "medium" for better accuracy
        print("✅ Whisper model loaded successfully!")
    return WHISPER_MODEL

@app.on_event("startup")
async def startup_load_whisper():
    """Load Whisper model when server starts"""
    try:
        load_whisper_model()
    except Exception as e:
        print(f"⚠️ Could not preload Whisper model: {e}")
        print("Model will be loaded on first transcription request")

#feedback function
@app.post("/api/message/feedback")
async def submit_message_feedback(request: Request):
    try:
        body = await request.json()
        
        # Get user email from cookie
        cookieHeader = request.headers.get("cookie") or ""
        emailMatch = cookieHeader.match(r"userEmail=([^;]+)")
        email = emailMatch.group(1) if emailMatch else None
        
        if not email:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Not authenticated"}
            )
        
        user = await async_get_or_create_user(email, email.split("@")[0])
        
        feedback_req = MessageFeedbackRequest(**body)
        
        # Validate feedback type
        if feedback_req.feedback_type not in ['good', 'bad']:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid feedback type"}
            )
        
        # Save feedback
        success = await async_save_message_feedback(
            user.id,
            feedback_req.message_id,
            feedback_req.feedback_type,
            feedback_req.comment
        )
        
        return JSONResponse(
            status_code=200,
            content={"success": success, "message": "Feedback saved"}
        )
        
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    

@app.post("/api/rag/upload")
async def upload_knowledge_pdf(file: UploadFile = File(...), email: str = Form(...)):
    """Upload a PDF to the RAG knowledge base"""
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"success": False, "error": "Only PDF files allowed"})
        
        os.makedirs("./knowledge_base", exist_ok=True)
        save_path = f"./knowledge_base/{file.filename}"
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        chunks = ingest_pdf(save_path)
        
        return {"success": True, "message": f"Ingested {chunks} chunks from {file.filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/rag/status")
async def rag_status():
    try:
        from rag import get_chunk_count
        count = get_chunk_count()
        return {"success": True, "total_chunks": count, "ready": count > 0}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    email: str = Form(...)
):
    """Transcribe audio to text using local Whisper model (FREE)"""
    try:
        print(f"🎤 Transcription request from {email}")
        
        # Verify user
        user = await async_get_or_create_user(email, email.split("@")[0])
        
        # Save uploaded file temporarily
        # Whisper works best with WAV or MP3 files
        file_extension = os.path.splitext(audio.filename)[1] or ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"📁 Temporary file saved: {temp_path} ({len(content)} bytes)")
        
        try:
            # Load Whisper model
            model = load_whisper_model()
            
            print("🔄 Transcribing audio...")
            
            # Transcribe using Whisper
            result = model.transcribe(
                temp_path,
                language=None,  # Auto-detect language 
                fp16=False,  # Set to True if you have a CUDA-enabled GPU for faster processing
                verbose=False
            )
            
            transcript_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            if not transcript_text:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "No speech detected. Please speak clearly and try again."
                    }
                )
            
            print(f"✅ Transcription successful (language: {detected_language}): {transcript_text[:100]}...")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "transcript": transcript_text,
                    "language": detected_language
                }
            )
        
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Failed to transcribe audio: {str(e)}"
                }
            )
        
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temporary file")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")
    
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
    
@app.get("/api/conversation/{user_email}/{conversation_id}/export")
async def export_conversation_pdf(user_email: str, conversation_id: str):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])
        
        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid conversation ID"}
            )
        
        # Get messages
        messages = await async_get_messages_for_conversation(conv_id_int)
        
        if not messages:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No messages found"}
            )
        
        # Get conversation metadata
        async with async_session_maker() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conv_id_int)
            )
            conv = result.scalars().first()
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='green',
            spaceAfter=30
        )
        story.append(Paragraph(f"Chat: {conv.title if conv else 'Conversation'}", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Messages
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=12,
            textColor='black',
            leftIndent=20
        )
        
        ai_style = ParagraphStyle(
            'AIMessage',
            parent=styles['Normal'],
            fontSize=12,
            textColor='darkgreen',
            leftIndent=20
        )
        
        for msg in messages:
            role_text = "You" if msg.role == "user" else "AI Agronomist"
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else ""
            
            style = user_style if msg.role == "user" else ai_style
            
            story.append(Paragraph(f"<b>{role_text}</b> - {timestamp}", styles['Heading3']))
            story.append(Paragraph(msg.content, style))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        buffer.seek(0)
        
        
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=chat_{conversation_id}.pdf"
            }
        )
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

#api for forget password

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

#api for verify reset code

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

#api for reset password

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

    try:
        from rag import init_vector_db, ingest_all_pdfs
        init_vector_db()   # creates the table in Neon if not exists
        ingest_all_pdfs()  # ingests any PDFs in knowledge_base/
    except Exception as e:
        print(f"⚠️ RAG init warning: {e}")


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

            for image_str in chat_req.images:
                try:
                    image_bytes = base64.b64decode(image_str)
                    await save_chat_image(user.id, db_conversation_id, image_bytes)
                except Exception as e:
                    print(f"❌ Image decode failed: {e}")
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
                        f"⚠️ The crop classifier was uncertain.\n\n"
                        f"{analysis['description']}\n\n"
                        f"{analysis['advice']}"
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
        # validate user exists.
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


# History & Conversation Endpoints 

@app.get("/api/history/{user_email}", response_model=HistoryResponse)
async def get_chat_history(user_email: str, limit: int = 20):  # Reduced limit
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        # Get conversations
        conversations = await async_get_conversations_for_user(user.id)
        conversations.sort(
            key=lambda x: x.get("last_message_at") or x.get("created_at") or "",
            reverse=True,
        )

        # Only get recent messages
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


@app.delete("/api/auth/delete-account")
async def delete_account(request: DeleteAccountRequest):
    """Permanently delete user account and archive all data"""
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                select(User).where(User.email == request.email)
            )
            user = result.scalars().first()

            if not user:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "error": "User not found"}
                )

            # Verify password before deletion
            if not verify_password(request.password, getattr(user, "hashed_password", "")):
                return JSONResponse(
                    status_code=401,
                    content={"success": False, "error": "Incorrect password"}
                )

            user_id = user.id

        success = await async_delete_account(user_id)

        if success:
            return {"success": True, "message": "Account deleted successfully"}
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to delete account"}
            )

    except Exception as e:
        print(f"❌ Delete account error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


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
