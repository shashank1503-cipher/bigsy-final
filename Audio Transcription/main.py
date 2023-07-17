from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils import getTranscription, download_data_from_FTP
import uvicorn
import os
# import socket
# hostname = socket.gethostname()
# IPAddr = socket.gethostbyname(hostname)
# print(IPAddr)
app = FastAPI()

@app.get("/")
async def root():
    routes = {
        "/": "This page",
        "/asr": "Get Sound transcription from the server",
    }
    return routes
@app.post("/asr")
async def asr(req: Request):
    data = await req.json()
    url = data['url']
    url = url.replace("127.0.0.1", "192.168.1.32") #need to get this at runtime
    print(url)
    try:
        file_path, file_name = download_data_from_FTP(url)
    except Exception as e:
        return {"meta": {"error": [str(e)]}, "data": {}}
    result = {
        "meta": {
            "file_name": file_name,
            "file_path": file_path,
            "error": []
        },
        "data" :{
            "transcription":"",
            "language": "Hindi"
        }
    }
    try:
        data = getTranscription(file_path, "Hindi", "lm")
        history, language = data['text'], data['language']
        result['data']['transcription'] = history['transcription']
        result['data']['language'] = language
        # result['data']['transcription'] = getTranscription(file_path, "Hindi", "lm")
    except Exception as e:
        result['meta']['error'].append(str(e))
    os.remove(file_path)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6500)