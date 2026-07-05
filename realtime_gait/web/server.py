"""FastAPI server for realtime gait web console."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .session import StreamSession

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Realtime Gait Console", version="1.0.0")
session = StreamSession()


class StreamStartRequest(BaseModel):
    url: str = ""


class EnrollStartRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    track_id: Optional[int] = None


class EnrollSelectTrackRequest(BaseModel):
    track_id: int = Field(..., ge=0)


class EnrollFinishRequest(BaseModel):
    allow_partial: bool = False


class GalleryPathRequest(BaseModel):
    path: str = ""


class GalleryDeleteRequest(BaseModel):
    person_id: str = Field(..., min_length=1, max_length=64)

class MessageResponse(BaseModel):
    ok: bool = True
    message: str = ""


def _ok(message: str) -> MessageResponse:
    return MessageResponse(ok=True, message=message)


def _run(action):
    try:
        msg = action()
        return _ok(msg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/status")
async def api_status():
    st = session.get_status()
    return {
        "running": st.running,
        "pipeline_ready": st.pipeline_ready,
        "default_stream": session.default_stream,
        "stream_url": st.stream_url,
        "mode": st.mode,
        "gallery_count": st.gallery_count,
        "gallery_path": st.gallery_path,
        "fps": st.fps,
        "lag_ms": st.lag_ms,
        "reader_seq": st.reader_seq,
        "reconnects": st.reconnects,
        "tracks": st.tracks,
        "timings_ms": st.timings_ms,
        "enroll": st.enroll,
        "message": st.message,
        "last_error": st.last_error,
        "recognition_enabled": st.recognition_enabled,
        "gallery_people": st.gallery_people,
        "frame_index": st.frame_index,
        "processed": st.processed,
    }


@app.post("/api/stream/start", response_model=MessageResponse)
async def api_stream_start(body: StreamStartRequest):
    return _run(lambda: session.start(body.url))


@app.post("/api/stream/stop", response_model=MessageResponse)
async def api_stream_stop():
    return _run(session.stop)


@app.post("/api/mode/preview", response_model=MessageResponse)
async def api_mode_preview():
    return _run(session.set_mode_preview)


@app.post("/api/mode/recognize", response_model=MessageResponse)
async def api_mode_recognize():
    return _run(session.set_mode_recognize)


@app.post("/api/enroll/start", response_model=MessageResponse)
async def api_enroll_start(body: EnrollStartRequest):
    return _run(lambda: session.start_enroll(body.name, body.track_id))


@app.post("/api/enroll/select", response_model=MessageResponse)
async def api_enroll_select(body: EnrollSelectTrackRequest):
    return _run(lambda: session.set_enroll_track(body.track_id))


@app.post("/api/enroll/finish", response_model=MessageResponse)
async def api_enroll_finish(body: EnrollFinishRequest):
    return _run(lambda: session.finish_enroll(allow_partial=body.allow_partial))


@app.post("/api/enroll/cancel", response_model=MessageResponse)
async def api_enroll_cancel():
    return _run(session.cancel_enroll)


@app.post("/api/gallery/save", response_model=MessageResponse)
async def api_gallery_save(body: GalleryPathRequest):
    return _run(lambda: session.save_gallery(body.path))


@app.post("/api/gallery/reload", response_model=MessageResponse)
async def api_gallery_reload(body: GalleryPathRequest):
    return _run(lambda: session.reload_gallery(body.path))


@app.post("/api/gallery/delete", response_model=MessageResponse)
async def api_gallery_delete(body: GalleryDeleteRequest):
    return _run(lambda: session.delete_gallery_person(body.person_id))

async def _mjpeg_generator() -> AsyncIterator[bytes]:
    boundary = b"--frame"
    while True:
        jpeg = session.get_jpeg()
        if jpeg is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.033)


@app.get("/api/video")
async def api_video():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def create_app(
    *,
    gallery_path: str = "output/gallery",
    default_stream: str = "",
    config_path: str = "",
) -> FastAPI:
    global session
    session = StreamSession(
        config_path=config_path,
        gallery_path=gallery_path,
        default_stream=default_stream or session._default_stream,
    )
    return app
