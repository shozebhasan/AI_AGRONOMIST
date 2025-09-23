import os
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, select
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# Force asyncpg driver
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session_maker = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


# -----------------------------
# Models
# -----------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    messages = relationship("Message", back_populates="user")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")


# -----------------------------
# Helpers
# -----------------------------
async def init_db():
    """Create tables if not exist"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def async_get_or_create_user(name: str):
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.name == name))
        user = result.scalars().first()
        if not user:
            user = User(name=name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


async def async_save_message(user_id: int, role: str, content: str):
    async with async_session_maker() as session:
        msg = Message(user_id=user_id, role=role, content=content)
        session.add(msg)
        await session.commit()
        await session.refresh(msg)
        return msg


async def async_get_history(user_id: int, limit: int = 10):
    async with async_session_maker() as session:
        result = await session.execute(
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))
