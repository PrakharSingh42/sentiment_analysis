from pydantic import BaseModel


class Review(BaseModel):
    cleaned_review: str