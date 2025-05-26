import os
import uuid
import time
import pandas as pd
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from agent_workflow import react_graph
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
os.makedirs("uploads", exist_ok=True)  # Create uploads directory
app.mount("/static", StaticFiles(directory="."), name="static")

# In-memory session storage
session_storage = {}

# Step 2: File Cleanup
def cleanup_files(background_tasks: BackgroundTasks):
    """Schedule cleanup of files older than 1 hour"""
    def delete_old_files():
        one_hour_ago = time.time() - 3600
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < one_hour_ago:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

    background_tasks.add_task(delete_old_files)

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    response = HTMLResponse(content=open("index.html", "r").read())
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/query")
async def process_query(
    request: Request,
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    file: UploadFile = File(None)
):
    logger.info(f"Received query: {query}")
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    
    # Get or initialize session data from in-memory storage
    session_data = session_storage.get(session_id, {"input_csv": None, "messages": []})
    input_csv = session_data["input_csv"]
    messages = session_data["messages"]

    # Convert stored messages to LangChain message objects
    converted_messages = []
    for msg in messages:
        if msg["type"] == "human":
            converted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "system":
            converted_messages.append(SystemMessage(content=msg["content"]))
        else:
            # For AI messages or others, keep as dictionary for now
            converted_messages.append(msg)

    # Add the new human message
    converted_messages.append(HumanMessage(content=query))

    input_file = None
    if file:
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            logger.error(f"File too large: {file.filename}")
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        try:
            file_extension = file.filename.split('.')[-1].lower()
            file_path = f"uploads/{uuid.uuid4()}.{file_extension}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            if file_extension in ['png', 'jpeg', 'jpg']:
                input_file = file_path
                logger.info(f"Image uploaded: {file_path}")
            elif file_extension == 'csv':
                try:
                    df = pd.read_csv(file_path)
                    if df.empty or len(df.columns) == 0:
                        logger.error(f"Invalid CSV: Empty or no columns in {file_path}")
                        raise HTTPException(status_code=400, detail="Invalid CSV file (empty or no columns)")
                except Exception as e:
                    logger.error(f"Invalid CSV format: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
                input_csv = file_path
                session_storage[session_id] = {"input_csv": input_csv, "messages": messages}
                logger.info(f"CSV uploaded: {file_path}")
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG, JPEG, or CSV")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error uploading file: {str(e)}")

    try:
        output = react_graph.invoke({
            "messages": converted_messages,
            "input_file": input_file,
            "input_csv": input_csv
        })
        logger.info(f"Graph output: {output}")
        
        # Extract the response
        if not output["messages"] or not hasattr(output["messages"][-1], "content"):
            logger.error("No valid response in output messages")
            raise HTTPException(status_code=500, detail="No valid response from agent")
        
        response = output["messages"][-1].content
        logger.info(f"Extracted response: {response}")
        
        # Store updated messages in in-memory storage
        messages = output["messages"]
        messages_json = [
            {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
            else {"type": "system", "content": m.content} if isinstance(m, SystemMessage)
            else {"type": "ai", "content": m.content}
            for m in messages
        ]
        session_storage[session_id] = {"input_csv": input_csv, "messages": messages_json}
        
        # Schedule file cleanup
        cleanup_files(background_tasks)
        
        return {"response": response, "loaded_csv": input_csv}
    except Exception as e:
        logger.error(f"Graph execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.post("/clear_csv")
async def clear_csv(request: Request, background_tasks: BackgroundTasks):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_storage:
        session_data = session_storage[session_id]
        input_csv = session_data["input_csv"]
        messages = session_data["messages"]
        if input_csv and os.path.exists(input_csv):
            try:
                os.remove(input_csv)
                logger.info(f"Deleted CSV file: {input_csv}")
            except Exception as e:
                logger.error(f"Error deleting CSV file {input_csv}: {str(e)}")
        session_storage[session_id] = {"input_csv": None, "messages": messages}
        logger.info(f"Cleared CSV for session: {session_id}")
    cleanup_files(background_tasks)
    return {"message": "CSV cleared"}

@app.post("/clear_session")
async def clear_session(request: Request, background_tasks: BackgroundTasks):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_storage:
        session_data = session_storage[session_id]
        input_csv = session_data["input_csv"]
        if input_csv and os.path.exists(input_csv):
            try:
                os.remove(input_csv)
                logger.info(f"Deleted CSV file: {input_csv}")
            except Exception as e:
                logger.error(f"Error deleting CSV file {input_csv}: {str(e)}")
        del session_storage[session_id]
        logger.info(f"Cleared session: {session_id}")
    cleanup_files(background_tasks)
    return {"message": "Session cleared"}