"""
FastAPI server for the Project Apricorn web visualizer.

All pipeline logic lives in apricorn/router_ui/session.py; this file only exposes it.
Endpoints: /api/board_state  /api/route_epoch  /api/run_pipeline  /api/load_board
           /api/upload_board /api/reset /api/model_metrics /api/eval_report /api/status
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apricorn.router_ui import session
from apricorn.router_ui.session import state  # re-exported: tests import `state` from here

app = FastAPI(title="Project Apricorn - PCB Routing Visualizer", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/board_state")
def get_board_state():
    return session.serialize_board_state()


@app.post("/api/route_epoch")
def route_epoch():
    state.step()
    return session.serialize_board_state()


class RunRequest(BaseModel):
    max_epochs: int = 15


@app.post("/api/run_pipeline")
def run_pipeline(req: RunRequest):
    state.run(req.max_epochs)
    return session.serialize_board_state()


class LoadBoardRequest(BaseModel):
    board_type: str = "demo_board.json"


@app.post("/api/load_board")
def load_board(req: LoadBoardRequest):
    state.load_preset(req.board_type)
    return session.serialize_board_state()


class UploadRequest(BaseModel):
    filename: str
    content: str


@app.post("/api/upload_board")
def upload_board(req: UploadRequest):
    try:
        state.load_uploaded(req.filename, req.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    return session.serialize_board_state()


@app.post("/api/reset")
def reset_board():
    state.reset()
    return session.serialize_board_state()


@app.get("/api/model_metrics")
def get_model_metrics():
    return session.model_metrics()


@app.get("/api/eval_report")
def get_eval_report():
    return session.eval_report()


@app.get("/api/status")
def get_status():
    return {"mode": session.model_mode(), "drc_engine": session.DRC_MODE,
            "checkpoint": str(session.CKPT_PATH), "checkpoint_exists": session.CKPT_PATH.exists()}


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Project Apricorn Server Running. Static index.html not yet created."}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apricorn.router_ui.server:app", host="0.0.0.0", port=8000, reload=False)
