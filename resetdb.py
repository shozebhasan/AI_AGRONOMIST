
import asyncio
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, select, desc, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# Neon connection
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Remove query parameters for clean connection
DATABASE_URL = DATABASE_URL.split('?')[0]

# Neon-optimized engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    connect_args={
        "ssl": "require",
    },
    poolclass=NullPool,
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

#  Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")

async def reset_database():
    """Completely reset the database with new schema"""
    print("🔄 Starting database reset...")
    
    try:
        # Drop all existing tables
        print("🗑️  Dropping existing tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("✅ All tables dropped successfully")
        
        # Create all tables with new schema
        print("🏗️  Creating new tables with updated schema...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ All tables created successfully")
        
        # Verify the schema using proper SQLAlchemy text()
        print("🔍 Verifying database schema...")
        async with engine.begin() as conn:
            # Check if tables exist using text() for raw SQL
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = result.scalars().all()
            print(f"📊 Found tables: {tables}")
            
            # Check if password_hash column exists in users table
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'password_hash'"
            ))
            password_column = result.scalar()
            if password_column:
                print("✅ password_hash column exists in users table")
            else:
                print("❌ password_hash column missing from users table")
        
        print("🎉 Database reset completed successfully!")
        print("📋 New schema includes:")
        print("   - users table with password_hash column")
        print("   - conversations table") 
        print("   - messages table")
        print("   - Proper relationships between tables")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database reset: {e}")
        return False

async def test_database():
    """Test the database connection and basic operations"""
    print("\n🧪 Testing database operations...")
    
    try:
        async with async_session_maker() as session:
            # Test user creation
            test_user = User(
                name="Test User",
                email="test@example.com",
                password_hash="test_hash"
            )
            session.add(test_user)
            await session.commit()
            await session.refresh(test_user)
            print("✅ User creation test passed")
            
            # Test user retrieval
            result = await session.execute(select(User).where(User.email == "test@example.com"))
            user = result.scalar()
            if user:
                print(f"✅ User retrieval test passed - User ID: {user.id}")
            else:
                print("❌ User retrieval test failed")
                
            # Clean up test user
            await session.delete(user)
            await session.commit()
            print("✅ Test cleanup completed")
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    return True

def main():
    """Main function to run the reset"""
    print("=" * 60)
    print("🛠️  AGNOMIST AI DATABASE RESET TOOL")
    print("=" * 60)
    print("⚠️  WARNING: This will DELETE ALL EXISTING DATA!")
    print("   Make sure you have backups if needed.")
    print("=" * 60)
    
    response = input("❓ Are you sure you want to reset the database? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\n🚀 Starting database reset process...")
        try:
            # Reset the database
            success = asyncio.run(reset_database())
            
            if success:
                # Test the database
                test_success = asyncio.run(test_database())
                
                print("\n" + "=" * 60)
                if test_success:
                    print("✅ DATABASE RESET COMPLETED SUCCESSFULLY!")
                    print("✅ Your authentication should now work properly.")
                else:
                    print("⚠️  Database reset completed but tests had issues.")
                print("✅ Restart your FastAPI server and test login/signup.")
                print("=" * 60)
            else:
                print("\n❌ DATABASE RESET FAILED")
                print("Please check your DATABASE_URL and try again.")
            
        except Exception as e:
            print(f"\n❌ DATABASE RESET FAILED: {e}")
            print("Please check your DATABASE_URL and try again.")
    else:
        print("❌ Database reset cancelled.")

if __name__ == "__main__":
    main()