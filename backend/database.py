import config
from motor.motor_asyncio import AsyncIOMotorClient

# MONGO_DETAILS = f"mongodb://{config.MONGO_DB_HOST}:{config.MONGO_DB_PORT}"
MONGO_DETAILS = f"mongodb://{config.MONGO_DB_HOST}:{config.MONGO_DB_PORT}/dpr-db"

db_client = AsyncIOMotorClient(MONGO_DETAILS)
db = db_client["dpr-db"]

