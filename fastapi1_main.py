import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from typing import List, Any, Optional, Union, Dict
from fopimt.task import Task, TaskConfig, TaskState, TaskInfo, TaskData
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fopimt import Magic
from fopimt.loader import Loader, ModulAPI, PackageType

import sentry_sdk

# How to run server:
# uvicorn fastapi_main:app --reload --port 8086

# HOW TO RUN WITH BETTER WORKING MULTIPROCESSING
# docker compose up
#   
# If changes were made to the docker setup, use --build
# docker compose up --build
#
# To run the docker in the background, use -d
# docker compose up -d


sentry_sdk.init(
    dsn="https://e9b23e443f8cee537da5d9a1875a4b96@o4508162086404096.ingest.de.sentry.io/4508170777329744",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    _experiments={
        # Set continuous_profiling_auto_start to True
        # to automatically start the profiler on when
        # possible.
        "continuous_profiling_auto_start": True,
    },
)

app = FastAPI(
    title="EASE",
    description="Effortless Algorithmic Solution Evolution (Fop-IMT)",
    version="1.0",
    contact={
        "name": "A.I.Lab",
        "url": "https://ailab.fai.utb.cz/"
    }
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specify which origins can access your API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including PATCH
    allow_headers=["*"],
)

magic_instance = Magic()

def _get_task(task_id: str) -> Task:
    task = magic_instance.task_get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task with the id {task_id} not exists.", )
    return task

###############################################
########## ENTRY POINTS
###############################################
# Landing page
# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=FileResponse)
def serve_landing_page():
    return "static/index.html"

# POST
# Create a new Task
# id: Optional[str] = None  - ID of the new Task, if empty string (None), Core will generate one
# return: TaskInfo          - Info of the created Task, or already existing one
@app.post("/task")
def task_create(task_id: Optional[str] = None) -> TaskInfo:
    _task = magic_instance.task_create(task_id)
    if _task is None:
        raise HTTPException(status_code=406, detail=f"Unable to create Task with the id {task_id}.", )
    return _task.get_info()

# GET
# Get info of all Tasks
# return: list[TaskInfo]  - Info of all existing tasks
@app.get("/task/all/info")
def task_info() -> list[TaskInfo]:
    tasks = magic_instance.task_get_all()
    task_info_list = list()
    for task in tasks:
        task_info_list.append(task.get_info())
    return task_info_list

# GET
# Get info of a Task
# task_id: str      - ID of the new Task, if empty string (None), Core will generate one
# return: TaskInfo  - Info of the created Task, or already existing one
@app.get("/task/{task_id}/info")
def task_info(task_id: str) -> TaskInfo:
    task = _get_task(task_id)
    return task.get_info()

# GET
# Get a list of ids of all Tasks in Core and their status code
# id: Optional[str] = None  - Specified task id, if None or non-existing, all task statuses are returned
# return: dict[str, int]    - dict of str(ID) of the Tasks and status codes
@app.get("/task/status", response_model=dict[str, TaskInfo])
def task_status(task_id: Optional[str] = None) -> dict[str, TaskInfo]:
    """
        Get the status of all tasks or a specific task.

        - `task_id` (Optional[str]): Specify a task ID to get the status of a single task.
          If not provided or non-existing, all task statuses will be returned.

        The response is a dictionary mapping each task ID to a status code:
        - `0` (CREATED): Task is created but not yet started or fully initialized.
        - `1` (INIT): Task is initialized and ready to be started.
        - `2` (RUN): Taskvis is currently running.
        - `3` (STOP): Task is stopped.
        - `4` (FINISH): Task has completed successfully.
        - `5` (BREAK): Task is interrupted or broken.
        """
    return magic_instance.get_task_info(task_id)


# GET
# Get a list of all Modul options for Task with specified ID
# id: str   - Specified task id
# return: list[Modul]    - List of all possible moduls with their type identified
@app.get("/task/{task_id}/options")
def task_options(task_id: str, task_configuration: TaskConfig) -> list[ModulAPI]:
    task = _get_task(task_id)
    out = []
    for _type in PackageType:
        out += magic_instance.get_loader().get_package(_type).get_moduls()
    return out

# GET
# Get a list of all Modul options for generic Task
# return: list[Modul]    - List of all possible moduls with their type identified
@app.get("/task/options")
def task_options_new() -> list[ModulAPI]:

    out = []
    for _type in PackageType:
        out += magic_instance.get_loader().get_package(_type).get_moduls()
    return out

# PUT
# Serves for initialization of the task
# Once the task is fully initialized (all required characteristics), state of the task changes to INIT
@app.put("/task/{task_id}")
def task_init(task_id: str, task_configuration: TaskConfig) -> TaskInfo:
    task = _get_task(task_id)
    try:
        task.initialize(loader=magic_instance.get_loader(), task_config=task_configuration)
    except Exception as e:
        raise HTTPException(status_code=406, detail=f"Unable to initialize Task with the id {task_id}. Error: {e}", )
    return task.get_info()

# POST
# Serves for duplication of the task
# TODO: since create endpoint offers asigning custom id to a task,
# I think this endpoint should offer it as well for consistancy.
@app.post("/task/{task_id}/duplicate")
def task_duplicate(task_id: str, new_name: str) -> TaskInfo:
    task = _get_task(task_id)
    new_task = magic_instance.task_duplicate(task, new_name)

    if new_task is None:
        raise HTTPException(status_code=500, detail=f"Unable to duplicate Task with the id {task_id}.", )

    return new_task.get_info()


# GET
# Get Task data (Messages and Solutions). All or from a set time.
# task_id: str  - ID of the Task
# dateFrom: str - Date from in str.# assuming it is in UTC format (ending with 'Z') "%Y-%m-%dT%H:%M:%S.%fZ"
# return: TODO  - description
@app.get("/task/data")
def get_tasks_data(request: Request, dateFrom: str | None = None) -> dict[str, TaskData]:
    news = magic_instance.get_new_data(dateFrom)

    for key in news.keys():
        tdata = news[key]

        for sol in tdata.solutions:
            if 'url' in sol.metadata.keys():
                sol.metadata['url'] = str(request.url_for("serve_image", filepath=sol.metadata['url']))

    return news

# GET
# Get Task solution by its message id.
# task_id: str      - ID of the Task
# message_id: str   - Message ID
# return: StreamingResponse
@app.get("/task/{task_id}/solution/{message_id}/download")
def task_get_solution(task_id: str, message_id: str) -> StreamingResponse:

    buffer = _get_task(task_id).get_solution(message_id)
    if buffer is None:
        raise HTTPException(status_code=500, detail=f"Could not create zip buffer for Task [{task_id}].")

    # Send the ZIP file as a StreamingResponse
    response = StreamingResponse(buffer, media_type='application/zip')
    response.headers["Content-Disposition"] = f"attachment; filename=archive_{task_id}_{message_id}.zip"

    return response


# GET
# Get whole Task as ZIP.
# task_id: str      - ID of the Task
# return: StreamingResponse
@app.get("/task/{task_id}/download")
def task_get_all(task_id: str) -> StreamingResponse:

    buffer = _get_task(task_id).get_all()
    if buffer is None:
        raise HTTPException(status_code=500, detail=f"Could not create zip buffer for Task [{task_id}].")

    # Send the ZIP file as a StreamingResponse
    response = StreamingResponse(buffer, media_type='application/zip')
    response.headers["Content-Disposition"] = f"attachment; filename=archive_{task_id}.zip"

    return response


# GET
# Get all LLM Connector types
# return: StreamingResponse
@app.get("/task/types/connectors", response_model=List[ModulAPI])
def get_connector_types() -> List[ModulAPI]:
    return magic_instance.get_llm_connectors()


# PATCH
# Run Task/Re-run paused task
# task_id: str  - ID of the Task
# return: bool  - True if start of the Task was succesfull, suceesfull, sucefull, ... OK
@app.patch("/task/{task_id}/run")
def task_run(task_id: str) -> TaskInfo:
    ret = magic_instance.task_run(task_id)
    if ret is False:
        logging.warning(f"Task [{task_id}] could not be run.")
    task = _get_task(task_id)
    return task.get_info()


# PATCH
# Pause Task
# task_id: str  - ID of the Task
# return: bool  - True if pause of the Task was OK
@app.patch("/task/{task_id}/pause")
def task_pause(task_id: str) -> TaskInfo:
    # TODO test more - so far works with docker
    ret = magic_instance.task_pause(task_id)
    if ret is False:
        logging.warning(f"Task [{task_id}] could not be paused.")
    task = _get_task(task_id)
    return task.get_info()


# PATCH
# Stop Task
# task_id: str  - ID of the Task
# return: bool  - True if stop of the Task was OK
@app.patch("/task/{task_id}/stop")
def task_stop(task_id: str) -> TaskInfo:
    #TODO test - not working
    ret = magic_instance.task_stop(task_id)
    if ret is False:
        logging.warning(f"Task [{task_id}] could not be stopped.")
    task = _get_task(task_id)
    return task.get_info()


# DELETE
# Delete Task. Only works for non-running tasks. Will essential rename the task folder, so it will be archived
# task_id: str  - ID of the Task
# return: bool  - True if delete (archivation) of the Task was OK
@app.delete("/task/{task_id}")
def task_delete(task_id: str) -> bool:
    task = _get_task(task_id)
    if not magic_instance.archive(task_id):
        return False
    return task.archive()

@app.get("/images/{filepath:path}")
async def serve_image(filepath: str):
    return FileResponse(filepath)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)
