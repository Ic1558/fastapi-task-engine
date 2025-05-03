from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os
import json
import aiohttp
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = FastAPI()

# === Config ===
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SHEET_ID = os.getenv("SHEET_ID", "").strip()
CREDENTIAL_PATH = "/etc/secrets/gdrive_sa.json"

# === Google API Helpers ===

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIAL_PATH,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def get_sheets_service():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIAL_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    return build("sheets", "v4", credentials=creds)

async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

async def upload_file_to_gdrive(file: UploadFile) -> str:
    service = get_drive_service()
    metadata = {"name": file.filename}
    if GDRIVE_FOLDER_ID:
        metadata["parents"] = [GDRIVE_FOLDER_ID]
    media = MediaIoBaseUpload(io.BytesIO(await file.read()), mimetype=file.content_type)
    uploaded = service.files().create(body=metadata, media_body=media, fields="id").execute()
    file_id = uploaded.get("id")
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def get_google_sheets_data():
    service = get_sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range="Sheet1!A1:C10"
    ).execute()
    return result.get("values", [])

# === Models ===

class UploadResponse(BaseModel):
    links: List[str]
    sheet_data: List[List[str]]

# === Endpoints ===

@app.get("/")
def root():
    return {"status": "FastAPI is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    try:
        links = []
        for file in files:
            link = await upload_file_to_gdrive(file)
            links.append(link)

        sheet_data = get_google_sheets_data()
        message = "📎 Upload Success:\n" + "\n".join(links)
        await send_telegram_message(message)
        return UploadResponse(links=links, sheet_data=sheet_data)
    except Exception as e:
        await send_telegram_message(f"❗ Upload Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sheets")
def read_sheet():
    return {"data": get_google_sheets_data()}

# === Local Execution ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
