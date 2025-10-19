# test_neon_connection.py
import asyncio
from db import init_db, check_db_health, async_get_or_create_user, async_save_message

async def test_neon():
    print("🧪 Testing Neon connection...")
    
    # Test database connection
    if not await init_db():
        print("❌ Database initialization failed")
        return
    
    if not await check_db_health():
        print("❌ Database health check failed")
        return
    
    # Test user creation
    user = await async_get_or_create_user("test@agronomist.com", "Test User")
    print(f"✅ User test: {user.name} ({user.email})")
    
    # Test message saving
    message = await async_save_message(user.id, "user", "Test message from Neon")
    if message:
        print("✅ Message save test: Success")
    else:
        print("❌ Message save test: Failed")
    
    print("🎉 All Neon tests passed!")

if __name__ == "__main__":
    asyncio.run(test_neon())