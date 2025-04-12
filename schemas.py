from pydantic import BaseModel
from typing import List
import operator
from typing_extensions import Annotated

class QueryResult(BaseModel):
    title: str
    url: str
    resume: str

class ReportState(BaseModel):
    user_input: str = None
    final_response: str = None
    queries : List[str] = []
    query_results : Annotated[List[QueryResult], operator.add]