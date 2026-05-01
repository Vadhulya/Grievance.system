from fastapi import APIRouter
from app.schemas.complaint import ComplaintRequest
from app.services.processor import process_complaint

router = APIRouter()

@router.post("/submit")
def submit_complaint(request: ComplaintRequest):
    return process_complaint(request.text)