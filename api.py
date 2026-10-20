from contextlib import contextmanager
import logging
import resource

# Captured before the heavy scientific/CMEMS stack is imported below, so the
# gap between this and the post-import reading isolates the import cost from
# a Render instance's memory cap (diagnosing repeated OOM restarts).
_pre_import_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from helpers import ControlAction, Location
from main import LOG_FORMAT, get_action, predict_surfacing, setup_logging, surface_trigger
from records import Records, ou_from_json, ou_to_json

setup_logging(debug=False)

logging.info("[memory] before heavy imports: %.1f MB, after: %.1f MB", _pre_import_rss_kb / 1024, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


app = FastAPI(
    title="EKF MPC Profiling float controller",
    description="API for making prediction of float",
    version="1.1.0",
)


@app.exception_handler(ValueError)
def value_error_handler(request, error: ValueError) -> JSONResponse:
    # e.g. no action stored for the cycle yet, fix with /update_stored_action
    logging.warning("%s failed: %s", request.url, error)
    return JSONResponse(status_code=HTTPStatus.BAD_REQUEST, content={"detail": str(error)})


def get_records(float_id: int) -> Records:
    try:
        return Records(float_id)
    except FileNotFoundError:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Float {float_id} not found")


@contextmanager
def float_log(records, name):
    """Also writes everything logged during the request to floats/<id>/logs/, fetch it with /get_file."""
    log_dir = records.float_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    # UTC like the log lines, copernicusmarine sets logging to UTC for the whole process
    handler = logging.FileHandler(log_dir / f"{name}_{datetime.now(timezone.utc):%Y-%m-%d-%H-%M-%S}Z.log")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    try:
        yield
    except Exception:
        logging.exception("%s for float %s failed", name, records.float_id)
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def state_to_dict(state) -> dict:
    return {
        "time": state.time.isoformat(),
        "latitude": state.location.latitude,
        "longitude": state.location.longitude,
        "depth": state.depth,
        "phase": state.phase,
        "bx": state.bx,
        "by": state.by,
    }


# Endpoints are plain def, not async, so the minutes-long simulations run in FastAPI's thread pool

@app.api_route("/", methods=["GET", "HEAD"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "message": "API is running and ready to accept requests."}

@app.post("/return_action")
def return_action(float_id: int, debug: bool) -> dict:
    with float_log(get_records(float_id), "return_action"):
        return asdict(get_action(float_id, debug))

@app.post("/update_state")
def update_state_endpoint(float_id: int, debug: bool) -> dict:
    with float_log(get_records(float_id), "update_state"):
        return state_to_dict(predict_surfacing(float_id, debug))

@app.post("/get_last_surfacing")
def get_last_surfacing(float_id: int) -> dict:
    records = get_records(float_id)
    last_cycle_number = records.get_last_cycle_number()
    if last_cycle_number is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No surfacings for float {float_id}")
    return records.get_surfacing_entry(last_cycle_number)

@app.post("/surfaced_update")
def surfaced_update(float_id: int, cycle_number: int, latitude: float, longitude: float, time: datetime, debug: bool) -> dict:
    with float_log(get_records(float_id), "surfaced_update"):
        return state_to_dict(surface_trigger(float_id, Location(latitude, longitude), time, cycle_number, debug))

@app.post("/update_stored_action")
def update_stored_action(float_id: int, cycle_number: int, drifting_depth: float, duration_hours: float) -> bool:
    records = get_records(float_id)
    last_cycle_number = records.get_last_cycle_number()
    if last_cycle_number is not None and cycle_number > last_cycle_number + 1:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Cycle {cycle_number} has not started yet, last surfacing is cycle {last_cycle_number}")

    records.save_action(ControlAction(drifting_depth, duration_hours), cycle_number, source="manual")
    return True

@app.post("/update_ou")
def update_ou(float_id: int, tau_e_hours: float = None, tau_n_hours: float = None, q_e: float = None, q_n: float = None) -> dict:
    """Sets the OU bias parameters used from the next simulation on, parameters left out keep their value.

    Every update is appended to floats/<id>/ou.log, fetch it with /get_file.
    """
    records = get_records(float_id)
    ou_json = ou_to_json(records.get_ou())
    changes = {name: value for name, value in dict(tau_e=tau_e_hours, tau_n=tau_n_hours, q_e=q_e, q_n=q_n).items() if value is not None}
    ou = ou_from_json({**ou_json, **changes})
    records.write_ou(ou, records.get_last_cycle_number() or 0)
    return ou_to_json(ou)

@app.post("/update_config")
def update_config(float_id: int, config: dict = Body(...)) -> bool:
    records = get_records(float_id)
    try:
        records.save_config(config)
    except KeyError as missing:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Config is missing {missing}")
    return True

@app.post("/get_errors_and_bias")
def get_errors_and_bias(float_id: int) -> PlainTextResponse:
    return PlainTextResponse(get_records(float_id).get_errors_and_bias(), media_type="text/csv")

@app.post("/get_file")
def get_file(float_id: int, file_name: str) -> JSONResponse:
    records = get_records(float_id)
    float_dir = records.float_dir.resolve()
    file_path = (float_dir / file_name).resolve()
    if float_dir not in file_path.parents:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid file_name {file_name}")
    if not file_path.is_file():
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"File {file_name} not found")

    file_contents = file_path.read_bytes()
    logging.info("Read file %s with size %d bytes", file_name, len(file_contents))
    return JSONResponse(content={"file_name": file_name, "file_contents": file_contents.decode("latin-1")})
