import os
import ssl
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    select,
    desc,
    func,
    delete,
    Boolean,
    LargeBinary
)
from sqlalchemy.pool import NullPool
from typing import Optional, List, Dict
from dotenv import load_dotenv
import hashlib

import secrets
from datetime import datetime, timedelta

load_dotenv()


# Database URL with asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

if "sslmode=" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?")[0]

# SSL context for Neon 
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Neon-optimized engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,   # set True for debugging
    future=True,
    poolclass=NullPool,
    connect_args={"ssl": ssl_context},
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


# Models

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    hashed_password = Column(String(255), nullable=True)
    memory = Column(Text, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(300), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    user = relationship("User", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, index=True)
    token = Column(String(6), nullable=False)  # 6-digit code
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)  # This uses sqlalchemy.Boolean
    created_at = Column(DateTime, default=datetime.utcnow)

#new

class UserFact(Base):
    __tablename__ = "user_facts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="facts")


class AddToMemory(Base):
    __tablename__ = "add_to_memory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="memory_entries")


class ChatImage(Base):
    __tablename__ = "chat_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    image_data = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="images")
    conversation = relationship("Conversation", backref="images")





# Password hashing helpers

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    return hash_password(plain_password) == hashed_password



# Database functions

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_health() -> bool:
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).limit(1))
            _ = result.scalars().first()
        return True
    except Exception as e:
        print(f"⚠️ DB health check failed: {e}")
        return False


async def async_get_or_create_user(email: str, name: str) -> User:
    """Fetch user by email, or create one without password (used for first-time chats)."""
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if user:
            return user

        new_user = User(email=email, name=name)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return new_user
    


async def async_create_conversation(user_id: int, title: str = "New Chat") -> Optional[int]:
    async with async_session_maker() as session:
        new_conv = Conversation(
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
        )
        session.add(new_conv)
        await session.commit()
        await session.refresh(new_conv)
        return new_conv.id


async def async_save_message(user_id: int, role: str, content: str, conversation_id: Optional[int] = None) -> Message:
    """Save a message and return the saved Message object."""
    async with async_session_maker() as session:
        msg = Message(
            user_id=user_id,
            role=role,
            content=content,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow(),
        )
        session.add(msg)

        # If conversation exists and has no title (New Chat), optionally update timestamps
        if conversation_id:
            await session.flush()
        await session.commit()
        await session.refresh(msg)
        return msg


async def async_get_history(user_id: int, limit: int = 50):
    """Optimized: Get recent messages directly without loading all conversations first"""
    async with async_session_maker() as session:
        # Just get the messages - much faster
        stmt = (select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.timestamp.desc())
                .limit(limit))
        
        result = await session.execute(stmt)
        messages = result.scalars().all()
        
        return list(reversed(messages))  # Return chronological order


# delete chats from sidebar

async def async_delete_conversation(conversation_id: int) -> bool:
    """
    Permanently delete conversation and its messages.
    Returns True if deleted, False if not found.
    """
    async with async_session_maker() as session:
        # Ensure conversation exists
        result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
        conv = result.scalars().first()
        if not conv:
            return False

        await session.execute(delete(ChatImage).where(ChatImage.conversation_id == conversation_id))

        # Delete messages explicitly (safe even if cascade configured)
        await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        # Delete conversation
        await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
        await session.commit()
        return True


async def async_get_conversations_for_user(user_id: int) -> List[dict]:
    """Optimized with composite index"""
    async with async_session_maker() as session:
        # This query now uses idx_conversations_user_created index
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.created_at))
            .limit(50)  # Add limit to avoid loading hundreds of conversations
        )
        convs = result.scalars().all()

        conversations = []
        for conv in convs:
            # This query now uses idx_messages_conversation_timestamp index
            stmt = select(func.count(Message.id), func.max(Message.timestamp)).where(
                Message.conversation_id == conv.id
            )
            r = await session.execute(stmt)
            count, last_ts = r.first() or (0, None)

            conversations.append({
                "id": conv.id,
                "title": conv.title or "New Chat",
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "message_count": int(count or 0),
                "last_message_at": last_ts.isoformat() if last_ts else None,
            })
        return conversations



async def async_get_messages_for_conversation(conversation_id: int, limit: int = 50) -> List[Message]:
    """Get messages with limit to avoid loading huge conversations"""
    async with async_session_maker() as session:
        stmt = (select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp.desc())
                .limit(limit))
        result = await session.execute(stmt)
        messages = list(result.scalars().all())
        return list(reversed(messages))
    

async def async_create_password_reset_token(email: str) -> Optional[str]:
    """Create a password reset token and return it"""
    async with async_session_maker() as session:
        # Delete any existing tokens for this email
        await session.execute(
            delete(PasswordResetToken).where(PasswordResetToken.email == email)
        )
        
        # Generate 6-digit code
        token = ''.join(secrets.choice('0123456789') for _ in range(6))
        expires_at = datetime.utcnow() + timedelta(minutes=15)  # 15 minutes expiry
        
        reset_token = PasswordResetToken(
            email=email,
            token=token,
            expires_at=expires_at
        )
        
        session.add(reset_token)
        await session.commit()
        return token

async def async_verify_password_reset_token(email: str, token: str) -> bool:
    """Verify if a password reset token is valid"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.email == email,
                PasswordResetToken.token == token,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.utcnow()
            )
        )
        reset_token = result.scalars().first()
        return reset_token is not None

async def async_use_password_reset_token(email: str, token: str) -> bool:
    """Mark a password reset token as used"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.email == email,
                PasswordResetToken.token == token,
                PasswordResetToken.used == False
            )
        )
        reset_token = result.scalars().first()
        
        if reset_token:
            reset_token.used = True
            await session.commit()
            return True
        return False

async def async_update_user_password(email: str, new_password: str) -> bool:
    """Update user's password"""
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        
        if user:
            user.hashed_password = hash_password(new_password)
            await session.commit()
            return True
        return False


async def save_user_fact(user_id: int, key: str, value: str):
    async with async_session_maker() as session:
        # Check if fact exists
        result = await session.execute(
            select(UserFact).where(UserFact.user_id == user_id, UserFact.key == key)
        )
        fact = result.scalars().first()
        if fact:
            fact.value = value
            fact.updated_at = datetime.utcnow()
        else:
            session.add(UserFact(user_id=user_id, key=key, value=value))
        await session.commit()

async def get_user_facts(user_id: int) -> Dict[str, str]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserFact.key, UserFact.value).where(UserFact.user_id == user_id)
        )
        return {key: value for key, value in result.all()}

async def add_memory_entry(user_id: int, content: str):
    async with async_session_maker() as session:
        session.add(AddToMemory(user_id=user_id, content=content))
        await session.commit()

        # Prune to last 15
        result = await session.execute(
            select(AddToMemory.id).where(AddToMemory.user_id == user_id).order_by(desc(AddToMemory.created_at))
        )
        ids = [r for r in result.scalars().all()]
        if len(ids) > 15:
            await session.execute(
                delete(AddToMemory).where(AddToMemory.id.not_in(ids[:6]))
            )
            await session.commit()

async def get_recent_memory(user_id: int) -> str:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AddToMemory.content).where(AddToMemory.user_id == user_id).order_by(desc(AddToMemory.created_at)).limit(15)
        )
        return "\n".join(reversed(result.scalars().all()))



async def save_chat_image(user_id: int, conversation_id: int, image_data: bytes, filename: str = "uploaded.jpg", content_type: str = "image/jpeg"):
    async with async_session_maker() as session:
        session.add(ChatImage(
            user_id=user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            image_data=image_data
        ))
        await session.commit()

