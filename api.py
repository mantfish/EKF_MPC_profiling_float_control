import logging
import resource

# Captured before the heavy scientific/CMEMS stack is imported below, so the
# gap between this and the post-import reading isolates the import cost from
# a Render instance's memory cap (diagnosing repeated OOM restarts).
_pre_import_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

import json
import os
import tempfile
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from main import surface_trigger
from main import update_state as run_update_state

from data_handler import *
from data_handler import DATA_ROOT, _float_dir, _log_memory
from helpers import Location, EstimatedState
from visulisation import build_visualization_html

logging.basicConfig(level=logging.INFO)

logging.info("[memory] before heavy imports: %.1f MB", _pre_import_rss_kb / 1024)
_log_memory("after heavy imports (app startup)")


def _parse_json_object(raw: bytes, field_name: str) -> dict:
    "Parses raw bytes as JSON, rejecting anything whose top level isn't an object."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid JSON in {field_name}: {e}"
        )
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"{field_name} must be a JSON object, got {type(parsed).__name__}",
        )
    return parsed


app = FastAPI(
    title="EKF MPC Profiling float controller",
    description="API for making prediction of float",
    version="1.0.0",
)


def _configure_float_logging(float_id: int, action_name: str) -> tuple[logging.Handler, Path]:
    "Adds a per-request FileHandler under the float's own log directory."
    log_dir = _float_dir(float_id) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{action_name}_{float_id}_on_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return handler, log_path


@app.api_route("/", methods=["GET", "HEAD"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "message": "API is running and ready to accept requests."}

@app.get("/visualize/{float_id}", response_class=HTMLResponse)
def visualize(float_id: int) -> HTMLResponse:
    if not check_if_float_exists(float_id):
        raise HTTPException(status_code=404, detail=f"Float {float_id} not found")
    return HTMLResponse(build_visualization_html(float_id))

@app.post("/return_action")
async def return_action(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid file format. Please upload a .json file."
        )

    contents = await file.read()
    float_id = _parse_json_object(contents, "file")["float_id"]
    does_float_exists = check_if_float_exists(float_id)
    if does_float_exists is False:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Float is not in database")

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
    does_float_exists = check_if_float_exists(float_id)
    if does_float_exists is False:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Float is not in database")
    handler, log_path = _configure_float_logging(float_id, "update_state")
    try:
        run_update_state(float_id)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return True

def initialise_float_core(config_json: dict, bathymetry_bytes: bytes, starting_contents: dict) -> None:
    """Writes a new float's config/bathymetry/initial state to disk and runs its first update_state.

    Shared by the /initialise_float endpoint and any bulk-onboarding scripts, so both
    go through the exact same validated logic.
    """
    float_id = config_json["float_id"]

    does_float_exists = check_if_float_exists(float_id)
    if does_float_exists is True:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"Float {float_id} is already in database, use update_state or return action instead",
        )

    # The bathymetry upload is always saved as bathymetry.nc (see below), so the
    # stored config should always agree — regardless of what the uploaded config
    # said (or omitted). Treat this as server-managed rather than client input.
    config_json["bathymetry_file_name"] = "bathymetry.nc"
    config_contents = json.dumps(config_json, indent=4).encode("utf-8")

    # Validate before touching disk: the action a float is initialised with must
    # be one of the actions listed in its own config, otherwise later simulation
    # steps (which look the action up by name) would fail on a bad float.
    possible_action_names = {a["name"] for a in config_json["possible_actions"]}
    starting_action_name = starting_contents["action_name"]
    if starting_action_name not in possible_action_names:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                f"Starting action '{starting_action_name}' is not one of this float's "
                f"possible_actions: {sorted(possible_action_names)}"
            ),
        )

    float_dir = _float_dir(float_id)
    float_dir.mkdir(parents=True, exist_ok=True)

    bathymetry_path = float_dir / "bathymetry.nc"
    with open(bathymetry_path, "wb") as f:
        f.write(bathymetry_bytes)

    config_path = float_dir / "config.json"
    with open(config_path, "wb") as f:
        f.write(config_contents)

    config = read_config(float_id)

    deploy_time = pd.Timestamp(starting_contents["time"]).tz_localize(None)
    deploy_location = Location(
        latitude=starting_contents["location"]["latitude"],
        longitude=starting_contents["location"]["longitude"],
    )
    x0, y0 = latlon_to_xy(deploy_location.latitude, deploy_location.longitude,
                        config.target_lat, config.target_lon)

    P0 = np.zeros((4, 4))
    P0[2:, 2:] = np.diag([1e-4, 1e-4])  # bias uncertainty prior; position known exactly

    deploy_state = EstimatedState(
        time=deploy_time,
        location=deploy_location,
        depth=starting_contents["depth"],
        phase=starting_contents["phase"],
        x=x0,
        y=y0,
        bx=0.0,
        by=0.0,
        P=P0,
    )

    write_surfacing_action_log(
        float_id=float_id,
        action_name=starting_contents["action_name"],
        surfaced_timestamp=deploy_time,
        surfaced_location=deploy_location,
        estimated_state=deploy_state,
        real_state=deploy_state,  # identical at t=0 — no innovation has happened yet
        nis=0.0,
        innovation=np.zeros(2),
        actions_cost={},
    )

    run_update_state(float_id)


@app.post("/initialise_float")
async def initialise_float_endpoint(
    config_file: UploadFile = File(...),
    bathymetry_file: UploadFile = File(...),
    starting_state_and_action: UploadFile = File(...),
) -> bool:
    if not config_file.filename or not config_file.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid file format for {config_file.filename}. Please upload a .json file."
        )

    if not starting_state_and_action.filename or not starting_state_and_action.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid file format for {starting_state_and_action.filename}. Please upload a .json file."
        )

    if not bathymetry_file.filename or not bathymetry_file.filename.endswith(".nc"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid file format for {bathymetry_file.filename}. Please upload a .nc file.",
        )

    config_json = _parse_json_object(await config_file.read(), "config_file")
    bathymetry_bytes = await bathymetry_file.read()
    starting_contents = _parse_json_object(await starting_state_and_action.read(), "starting_state_and_action")

    initialise_float_core(config_json, bathymetry_bytes, starting_contents)

    return True

@app.post("/update_config")
async def update_config(config_file: UploadFile = File(...)) -> bool:

    if not config_file.filename or not config_file.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid file format for {config_file.filename}. Please upload a .json file.",
        )

    config_contents = await config_file.read()
    config_json = _parse_json_object(config_contents, "config_file")
    float_id = config_json["float_id"]

    if not check_if_float_exists(float_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Float {float_id} not found. Use initialise_float first.",
        )

    # bathymetry.nc is never re-uploaded here, so keep this server-managed
    # field consistent with the file already on disk (see initialise_float_endpoint).
    config_json["bathymetry_file_name"] = "bathymetry.nc"
    config_contents = json.dumps(config_json, indent=4).encode("utf-8")

    # Validate before touching disk — raises if malformed/missing keys
    config_from_dict(config_json, float_id)

    config_path = _float_dir(float_id) / "config.json"
    with open(config_path, "wb") as f:
        f.write(config_contents)

    return True

@app.post("/get_last_surfacing")
async def get_last_surfacing(float_id: int) -> dict:
    if not check_if_float_exists(float_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Float {float_id} not found.",
        )

    last_surfacing = read_last_surfacing(float_id)
    if last_surfacing is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No surfacing data found for float {float_id}.",
        )

    return last_surfacing


@app.post("/get_file")
async def get_file(file_name: str):
    file_path = _resolve_within_data_root(file_name)
    if not file_path.exists():
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"File {file_name} not found.",
        )
    try:
        with open(file_path, "rb") as f:
            file_contents = f.read()
            logging.info(f"Read file {file_name} with size {len(file_contents)} bytes")
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error reading file {file_name}: {str(e)}, probably does not exist",
        )


    return JSONResponse(content={"file_name": file_name, "file_contents": file_contents.decode("latin-1")})

@app.post("/update_file")
async def update_file(float_id: int, file: UploadFile = File(...), file_name: str = None, are_you_sure: bool = False) -> bool:
    logging.info(f"Received request to update file {file.filename} for float {float_id}")
    if are_you_sure is False:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Are you sure you want to update this file?",
        )

    if not check_if_float_exists(float_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Float {float_id} not found.",
        )

    target_name = file_name or file.filename
    if not target_name or Path(target_name).name != target_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid file_name {target_name}.",
        )
    old_file_path = _float_dir(float_id) / target_name

    if old_file_path.is_file():
        with open(old_file_path, "wb") as old_file:
            old_file.write(await file.read())
        logging.info(f"Updated file {target_name} for float {float_id}")
    else:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"File {target_name} not found for float {float_id}.",
        )
    return True