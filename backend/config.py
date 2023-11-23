import os

from dotenv import load_dotenv

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", None) 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MongoDB Configuration
# MONGO_DB_PORT = os.getenv("MONGO_DB_PORT", None)
# MONGO_DB_HOST = os.getenv("MONGO_DB_HOST", None)
MONGO_DB_PORT = os.getenv("MONGO_DB_PORT", "27017")
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST", "localhost")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", None)
SERVER_PORT = os.getenv("SERVER_PORT", None)

# Constants and Global Variables
# MODEL_DIR = "model/"
MODEL_DIR = "model/"
# USER_DIR = f"users/{userid}"
