from pydantic import BaseModel
from typing import List, Optional

class QueryResult(BaseModel):
    title: str
    url: str
    resume: str

class ReportState(BaseModel):
    user_input: str = None
    final_response: str = None
    queries : List[str] = []