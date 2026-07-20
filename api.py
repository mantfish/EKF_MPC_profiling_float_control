from datetime import datetime
from http import HTTPStatus
from importlib.resources import path
import logging
from fastapi import FastAPI
from main import update_state, surface_trigger


app = FastAPI(
    title="EKF MPC Profiling float controller",
    description="API for making prediction of float",
    version="1.0.0",
)


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "API is running and ready to accept requests."}

@app.post("/return_action")
async def return_action(file: UploadFile = File(...)) -> JSONResponse:
    logging_filename = path.Path.join(path.dirname(float_id + "/logs"), f"surface_trigger_{float_id}_on_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    logging.basicConfig(filename=logging_filename, level=logging.INFO)
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid file format. Please upload a .json file."
        )
    control_action = surface_trigger(file)


@app.post("/update_state")
async def update_state(float_id: int) -> bool:
    logging_filename = path.Path.join(path.dirname(float_id + "/logs"), f"update_state_{float_id}_on_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    logging.basicConfig(filename=logging_filename, level=logging.INFO)
    update_state(float_id)




