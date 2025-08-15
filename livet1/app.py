from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from yolov8_camera import generate_frames
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Two-Wheeler Parking Detection</title>
        </head>
        <body style="text-align: center;">
            <h1>Live Parking Camera Feed</h1>
            <img src="/video" width="800" />
        </body>
    </html>
    """

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# 🔥 Running command inside app.py
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
