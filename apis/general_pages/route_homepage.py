# route_homepage.py
# ref: https://www.fastapitutorial.com/blog/serving-html-fastapi/
import shutil

from fastapi import APIRouter, Path
from fastapi import Request
from fastapi.templating import Jinja2Templates
from typing import Annotated

from fastapi import FastAPI, File, UploadFile

templates = Jinja2Templates(directory="templates")
general_pages_router = APIRouter()


@general_pages_router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("general_pages/homepage.html", {"request": request})


@general_pages_router.post("/inference")
async def inference(file: UploadFile = File(...)):
    if not file:
        return {'message': 'No upload file sent'}
    else:
        path = Path('/tmp') / file.filename
        size = path.write_bytes(await file.read())
        return {'file': path, 'bytes': size}
