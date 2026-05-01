from fastapi import FastAPI
from app.api import complaint

app = FastAPI()

app.include_router(complaint.router, prefix="/complaints")

@app.get("/")
def root():
    return {"message": "Grievance AI Backend Running"}
