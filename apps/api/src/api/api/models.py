from pydantic import BaseModel, Field
from typing import Optional 

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")


class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item. If no image is available, return None.")
    price: Optional[float] = Field(None,description="The price of the item")
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(..., description="Information about the items used to answer the query")