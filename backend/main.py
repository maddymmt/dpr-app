import glob
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, List, Optional

import uvicorn
from auth import create_access_token, get_password_hash, verify_password
from bson import ObjectId
from config import (ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, MODEL_DIR,
                    SECRET_KEY, SERVER_HOST, SERVER_PORT)
from database import db
from fastapi import (APIRouter, Depends, FastAPI, File, HTTPException,
                     UploadFile, status)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader, PreProcessor
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import convert_files_to_docs
from models import Document, Token, TokenData, User
from routes import create_user_route
from routes import router as user_router
from utils import clear_directory, get_user_dir
from pymongo.errors import DuplicateKeyError

# Initialize Polish reader
# READER_PL = FARMReader("henryk/bert-base-multilingual-cased-finetuned-polish-squad2", context_window_size=1500)
READER_PL = FARMReader("henryk/bert-base-multilingual-cased-finetuned-polish-squad2", context_window_size=1500, use_gpu=True)

app = FastAPI()

# app.include_router(user_router, prefix="/api/v1", tags=["users", "documents"])


# def get_user_dir(userid):
#     # Function to format the USER_DIR path based on the user ID
#     return Path(f"users/{userid}")

# def clear_directory(path):
#     for file in Path(path).glob('*'):
#         file.unlink()

def setup_document_store(userid, split_length):
    # user_dir_path = Path(USER_DIR.format(userid=userid))
    user_dir_path = get_user_dir(userid)

    db_path = user_dir_path / f"{userid}_PL_index.db"
    doc_store = FAISSDocumentStore(sql_url=f"sqlite:///{db_path}")
    
    doc_dir = user_dir_path / "documentspreproces"
    all_docs = convert_files_to_docs(dir_path=doc_dir)

    retriever = DensePassageRetriever.load(load_dir=MODEL_DIR, document_store=doc_store, use_gpu=True)
    
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=split_length,
        split_respect_sentence_boundary=True,
    )

    docs = preprocessor.process(all_docs)
    doc_store.write_documents(docs)
    doc_store.update_embeddings(retriever=retriever)
    index_path = user_dir_path / f"{userid}_PL_faiss_index.faiss"
    config_path = user_dir_path / f"{userid}_PL_faiss_index.json"
    doc_store.save(index_path=index_path, config_path=config_path)


@app.post("/users/", response_model=User)
async def create_user(user: User) -> User:
    if user.password is None:
        raise HTTPException(status_code=400, detail="Password is required")
    
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict(by_alias=True)
    user_dict.pop('_id', None)
    user_dict.pop('password', None)
    user_dict['hashed_password'] = hashed_password
    print("Username:", user_dict.get('username'))
    print("Email:", user_dict.get('email'))

    try:
        await db["users"].create_index([("username", 1)], unique=True)
        await db["users"].create_index([("email", 1)], unique=True)
        new_user = await db["users"].insert_one(user_dict)
    except DuplicateKeyError as e:
        # Check the error details to identify which field caused the duplication
        error_msg = str(e)
        print(error_msg)
        # print(user_dict['nickname'])

        detail = "Username or email already exists."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

    # Once successfully inserted, retrieve the new user
    created_user = await db["users"].find_one({"_id": new_user.inserted_id})
    # Do not include hashed_password in the response
    created_user.pop('hashed_password', None)
    
    return User(**created_user)

# Retrieve user by username route
@app.get("/users/{username}", response_model=User)
async def get_user_by_username(username: str) -> User:
    user = await db["users"].find_one({"username": username})
    if user:
        return User(**user)
    raise HTTPException(status_code=404, detail="User not found")

# Document creation route
@app.post("/documents/", response_model=Document)
async def create_document(user_id: str, document: Document) -> Document:
    document.owner_id = ObjectId(user_id)
    new_document = await db["documents"].insert_one(document.dict(by_alias=True))
    created_document = await db["documents"].find_one({"_id": new_document.inserted_id})
    return Document(**created_document)

# Retrieve documents for a user route
@app.get("/documents/{user_id}", response_model=List[Document])
async def get_documents_for_user(user_id: str) -> List[Document]:
    documents = await db["documents"].find({"owner_id": ObjectId(user_id)}).to_list(None)
    return [Document(**doc) for doc in documents]


# Authentication endpoint for generating token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db["users"].find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user.get("hashed_password")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get('/loadingdocstores')
async def load_doc_stores(userid, split_length=1000):
    setup_document_store(userid, int(split_length))
    return {"status": "Document store loaded"}


@app.post('/uploadfiles')
async def upload_files(userid: str, file: UploadFile):
    user_dir_path = get_user_dir(userid)
    # user_dir_path = Path(USER_DIR.format(userid=userid))
    doc_dir = user_dir_path / "documentspreproces"
    doc_dir.mkdir(parents=True, exist_ok=True)

    file_location = doc_dir / file.filename
    with open(file_location, "wb") as f:
        f.write(await file.read())

    return {"status": "File uploaded successfully", "filename": file.filename}

@app.get('/deletefiles')
async def delete_files(userid):
    clear_directory(get_user_dir(userid) / "documentspreproces")

    # clear_directory(Path(USER_DIR).format(userid=userid) / "documentspreproces")
    return {"status": "Files deleted"}

def run_query_pipeline(userid, query, top_k_Retr, top_k_Read):
    save_dir = MODEL_DIR
    # index_path = Path(USER_DIR.format(userid=userid)) / f"{userid}_PL_faiss_index.faiss"
    # config_path = Path(USER_DIR.format(userid=userid)) / f"{userid}_PL_faiss_index.json"
    index_path = get_user_dir(userid) / f"{userid}_PL_faiss_index.faiss"
    config_path = get_user_dir(userid) / f"{userid}_PL_faiss_index.json"
    pipeline = ExtractiveQAPipeline(
        reader=READER_PL, 
        retriever=DensePassageRetriever.load(
            save_dir, 
            FAISSDocumentStore.load(index_path=index_path, config_path=config_path)
        )
    )
    result = pipeline.run(query, params={"Retriever": {"top_k": top_k_Retr}, "Reader": {"top_k": top_k_Read}})
    return result

@app.get('/queryPL')
async def query(userid, q, top_k_Retr, top_k_Read):
    return run_query_pipeline(
        userid=userid, 
        query=q, 
        top_k_Read=int(top_k_Read), 
        top_k_Retr=int(top_k_Retr)
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=SERVER_HOST, port=int(SERVER_PORT), reload=True)