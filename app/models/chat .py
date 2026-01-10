from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ChatBody(BaseModel):
    message: str
    messages: Optional[List[Dict[str, Any]]] = None

