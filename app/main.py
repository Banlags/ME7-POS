import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import vision
from . import state

# Create FastAPI app instance
app = FastAPI(title="Visual POS System")

# Determine base directory for template and static paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files (CSS, JS, sounds)
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set up Jinja2 templates
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)


@app.on_event("startup")
async def startup_event():
    """
    Called when FastAPI starts.
    Start the camera + YOLO + MediaPipe capture loop.
    """
    print("[APP] Startup event: starting camera.")
    vision.start_camera()


@app.on_event("shutdown")
async def shutdown_event():
    """
    Called when FastAPI shuts down.
    Signal the camera thread to stop.
    """
    print("[APP] Shutdown event: stopping camera.")
    vision.stop_camera()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Main POS page.
    Left: camera feed.
    Right: POS receipt and totals.
    The dynamic data will come via /pos_state polling.
    """
    context = {
        "request": request,
        "session_status": "WAITING FOR SESSION",
        "items": [],
        "total_amount": 0.0,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/video_feed")
async def video_feed():
    """
    MJPEG video stream endpoint.
    The browser uses this as the src of an <img> tag.
    """

    def frame_generator():
        while True:
            frame_bytes = vision.get_latest_frame_jpeg()
            if frame_bytes is not None:
                # multipart/x-mixed-replace format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            else:
                # No frame yet; small sleep to avoid busy-wait
                time.sleep(0.05)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/pos_state")
async def pos_state():
    """
    Return the current POS session state as JSON:
    - session_active
    - session_status
    - items (list)
    - total_amount
    - last_session_total
    - last_session_item_count
    """
    return JSONResponse(content=state.get_pos_state_dict())