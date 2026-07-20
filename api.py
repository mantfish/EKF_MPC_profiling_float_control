import json
import logging
import os
import tempfile
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from main import surface_trigger
from main import update_state as run_update_state

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EKF MPC Profiling float controller",
    description="API for making prediction of float",
    version="1.0.0",
)


def _configure_float_logging(float_id: int, action_name: str) -> tuple[logging.Handler, Path]:
    "Adds a per-request FileHandler under the float's own log directory."
    log_dir = Path(f"float_{float_id}") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{action_name}_{float_id}_on_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return handler, log_path


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "API is running and ready to accept requests."}

@app.post("/return_action")
async def return_action(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid file format. Please upload a .json file."
        )

    contents = await file.read()
    float_id = json.loads(contents)["float_id"]

    handler, log_path = _configure_float_logging(float_id, "surface_trigger")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        selected_action_name = surface_trigger(tmp_path)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
        logging.getLogger().removeHandler(handler)
        handler.close()

    return JSONResponse(content={"action": selected_action_name})

@app.post("/update_state")
async def update_state_endpoint(float_id: int) -> bool:
    handler, log_path = _configure_float_logging(float_id, "update_state")
    try:
        run_update_state(float_id)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return True
