from fastapi import FastAPI
from app.ui import main as run_app

app = FastAPI()

@app.get("/")
def start():
    run_app()
    return {"message": "Self Driving App Running"}