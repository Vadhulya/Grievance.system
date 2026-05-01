from pydantic import BaseModel

class ComplaintRequest(BaseModel):
    text: str