import os

from typing import TypedDict
class State(TypedDict):
    topic: str
    summary: str
    score: int


from pydantic import BaseModel, field_validator
class StateModel(BaseModel):
    topic: str
    summary: str = ""
    score: int

    @field_validator("score")
    def validate_score(cls, value):
        if not (0 <= value <= 100):
            raise ValueError("Score must be between 0 and 100")
        return value

from dataclasses import dataclass, field
@dataclass
class StateDataclass:
    topic: str = ""
    summary: str = ""
    message: list[str] = field(default_factory=list)

from langgraph.graph import MessagesState
class StateMessages(MessagesState):
    user_name: str = ""
    language: str = ""
