from fastapi import FastAPI, HTTPException, UploadFile, File
import redis
import json
import uuid
import datetime
import requests
import aiohttp
from typing import Dict, List
from dotenv import load_dotenv
import os
from ratelimit import limits, sleep_and_retry
from retrying import retry
from agents.gg import dispatch_task
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import asyncio

load_dotenv()
app = FastAPI()
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

API_KEYS = {
    "GG": os.getenv("GG_API_KEY"),
    "Lisa": os.getenv("LISA_API_KEY"),
    "Claude": os.getenv("CLAUDE_API_KEY"),
    "EdgePM": os.getenv("EDGEPM_API_KEY"),
    "TradeBot": os.getenv("TRADEBOT_API_KEY")
}

# Google API Config
CREDENTIAL_PATH = "/etc/secrets/gdrive_sa.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

CALLS_PER_MINUTE = 60

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(CREDENTIAL_PATH, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def get_sheets_service():
    creds = service_account.Credentials.from_service_account_file(CREDENTIAL_PATH, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds)

@retry(stop_max_attempt_number=3, wait_fixed=2000)
async def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise HTTPException(status_code=500, detail="Telegram Token or Chat ID not configured")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.post(url, json=payload, timeout=10) as response:
                result = await response.json()
                log_to_markdown(
                    task="Telegram Push",
                    status="Attempted",
                    details={"message": message, "response": result},
                    anchor_ref=f"Grok_{str(datetime.datetime.now())}"
                )
                if not result.get("ok"):
                    error_message = result.get("description", "Unknown Telegram Error")
                    raise HTTPException(status_code=500, detail=f"Telegram Error: {error_message}")
                return result
    except Exception as e:
        log_to_markdown(
            task="Telegram Push",
            status="Failed",
            details={"message": message, "error": str(e)},
            anchor_ref=f"Grok_{str(datetime.datetime.now())}"
        )
        raise HTTPException(status_code=500, detail=f"Telegram Push Failed: {str(e)}")

@retry(stop_max_attempt_number=3, wait_fixed=2000)
async def upload_file_to_gdrive(file: UploadFile) -> str:
    service = get_drive_service()
    metadata = {"name": file.filename}
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaIoBaseUpload(io.BytesIO(await file.read()), mimetype=file.content_type)
    uploaded = service.files().create(body=metadata, media_body=media, fields="id").execute()
    file_id = uploaded.get("id")
    return f"https://drive.google.com/uc?id={file_id}&export=download"

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_google_sheets_data():
    service = get_sheets_service()
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="Sheet1!A1:C10"
    ).execute()
    return result.get("values", [])

def log_to_markdown(task: str, status: str, details: dict, anchor_ref: str):
    log_entry = {
        "task": task,
        "status": status,
        "details": details,
        "timestamp": str(datetime.datetime.now()),
        "anchor_ref": anchor_ref
    }
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"/Logs/_sync/FRAME01_REF_{date_str}.md"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(f"# AI Debate Sync Log ({date_str})\n\n")
    with open(log_file, "a") as f:
        f.write(f"## {log_entry['timestamp']}\n")
        f.write(f"- **Task**: {log_entry['task']}\n")
        f.write(f"- **Status**: {log_entry['status']}\n")
        f.write(f"- **Details**: {json.dumps(log_entry['details'])}\n")
        f.write(f"- **Anchor Ref**: {log_entry['anchor_ref']}\n\n")
    return log_entry

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def call_gpt_api(agent: str, command: str, api_key: str) -> Dict:
    response = requests.post(
        "https://api.telegram.org/bot{token}/sendMessage",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"You are {agent}, performing the task: {command}"},
                {"role": "user", "content": command}
            ],
            "temperature": 0.7
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def record_metrics(agent: str, task_id: str, response_time: float, success: bool):
    metrics = {
        "agent": agent,
        "task_id": task_id,
        "response_time": response_time,
        "success": success,
        "error": not success,
        "timestamp": str(datetime.datetime.now())
    }
    redis_client.lpush("metrics", json.dumps(metrics))
    log_to_markdown(
        task="Metrics",
        status="Recorded",
        details=metrics,
        anchor_ref=f"Grok_{str(datetime.datetime.now())}"
    )

def get_specialist_queue_size(domain: str) -> int:
    return redis_client.llen(f"task_queue_{domain}")

class UploadResponse(BaseModel):
    links: List[str]
    sheet_data: List[List[str]]

@app.get("/")
async def root():
    return {"status": "FastAPI is running"}

@app.post("/dispatch")
async def dispatch_task_endpoint(recipient: str, command: str, payload: dict):
    message = dispatch_task(recipient, command, payload)
    return {"status": "dispatched", "message_id": message["message_id"]}

@app.post("/lisa/upload", response_model=UploadResponse)
async def lisa_upload(files: List[UploadFile] = File(...)):
    start_time = time.time()
    try:
        links = []
        for file in files:
            link = await upload_file_to_gdrive(file)
            links.append(link)

        sheet_data = get_google_sheets_data()
        message = "📎 Upload Success:\n" + "\n".join(links)
        await send_telegram_message(message)  # ใช้ await เพื่อให้ FastAPI รอการส่งข้อความ
        response_time = time.time() - start_time
        record_metrics("Lisa", payload.get("task_id", "unknown"), response_time, True)
        log_to_markdown(
            task="Lisa_Upload_Doc",
            status="Processed",
            details={"links": links, "sheet_data": sheet_data},
            anchor_ref=f"Grok_{str(datetime.datetime.now())}"
        )
        return UploadResponse(links=links, sheet_data=sheet_data)
    except Exception as e:
        response_time = time.time() - start_time
        record_metrics("Lisa", payload.get("task_id", "unknown"), response_time, False)
        try:
            await send_telegram_message(f"❗ Lisa Upload Failed: {str(e)}")
        except Exception as telegram_error:
            log_to_markdown(
                task="Telegram Push During Error",
                status="Failed",
                details={"message": f"❗ Lisa Upload Failed: {str(e)}", "telegram_error": str(telegram_error)},
                anchor_ref=f"Grok_{str(datetime.datetime.now())}"
            )
        raise HTTPException(status_code=500, detail=f"Error in Lisa: {str(e)}")

@app.post("/claude/check_logic")
async def claude_check_logic(data: str):
    start_time = time.time()
    try:
        command = f"Check logic consistency of: {data}"
        response = call_gpt_api("Claude", command, API_KEYS["Claude"])
        response_time = time.time() - start_time
        record_metrics("Claude", payload.get("task_id", "unknown"), response_time, True)
        log_to_markdown(
            task="Claude_Check_Logic",
            status="Processed",
            details={"data": data, "response": response},
            anchor_ref=f"Grok_{str(datetime.datetime.now())}"
        )
        return {"status": "processed", "response": response}
    except Exception as e:
        response_time = time.time() - start_time
        record_metrics("Claude", payload.get("task_id", "unknown"), response_time, False)
        await send_telegram_message(f"❗ Claude Check Logic Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Claude: {str(e)}")

@app.post("/edgepm/process_edge_task")
async def edgepm_process_task(task_data: str):
    start_time = time.time()
    try:
        command = f"Process EDGE task: {task_data}"
        response = call_gpt_api("EdgePM", command, API_KEYS["EdgePM"])
        response_time = time.time() - start_time
        record_metrics("EdgePM", payload.get("task_id", "unknown"), response_time, True)
        log_to_markdown(
            task="EdgePM_Process_Task",
            status="Processed",
            details={"task_data": task_data, "response": response},
            anchor_ref=f"Grok_{str(datetime.datetime.now())}"
        )
        return {"status": "processed", "response": response}
    except Exception as e:
        response_time = time.time() - start_time
        record_metrics("EdgePM", payload.get("task_id", "unknown"), response_time, False)
        await send_telegram_message(f"❗ EdgePM Process Task Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in EdgePM: {str(e)}")

@app.post("/tradebot/process_trade_task")
async def tradebot_process_task(trade_data: str):
    start_time = time.time()
    try:
        command = f"Process TRADE task: {trade_data}"
        response = call_gpt_api("TradeBot", command, API_KEYS["TradeBot"])
        response_time = time.time() - start_time
        record_metrics("TradeBot", payload.get("task_id", "unknown"), response_time, True)
        log_to_markdown(
            task="TradeBot_Process_Task",
            status="Processed",
            details={"trade_data": trade_data, "response": response},
            anchor_ref=f"Grok_{str(datetime.datetime.now())}"
        )
        return {"status": "processed", "response": response}
    except Exception as e:
        response_time = time.time() - start_time
        record_metrics("TradeBot", payload.get("task_id", "unknown"), response_time, False)
        await send_telegram_message(f"❗ TradeBot Process Task Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in TradeBot: {str(e)}")

@app.get("/dashboard")
async def dashboard():
    task_queue_size = redis_client.llen("task_queue")
    failed_queue_size = redis_client.llen("failed_task_queue")
    activity_logs = redis_client.lrange("activity_logs", 0, 9)
    activity_logs = [json.loads(log) for log in activity_logs]
    dashboard_data = {
        "task_queue_size": task_queue_size,
        "failed_queue_size": failed_queue_size,
        "recent_logs": activity_logs
    }
    log_to_markdown(
        task="Dashboard",
        status="Accessed",
        details=dashboard_data,
        anchor_ref=f"Grok_{str(datetime.datetime.now())}"
    )
    return dashboard_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
