from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings

import json
import os
from pydantic import BaseModel
from typing import List, Optional

class TopicRequest(BaseModel):
    newtopic: str
    description: str
    
class EmailEntry(BaseModel):
    content:str
    groundtruth: Optional[str]=None

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ML Server with Feature Generation Factory",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

#part 2: endpoint for new topics
@app.post("/api/v1/topics/add")
def addnewtopic(request: TopicRequest):
    with open('/home/ec2-user/environment/lab2_factories/data/topic_keywords.json','r') as f:
        data=json.load(f)
        
    if request.newtopic not in data:
        data[request.newtopic]={
            'description': request.description
        }
        
        
        with open('/home/ec2-user/environment/lab2_factories/data/topic_keywords.json', 'w') as f:
            json.dump(data, f)
        return {"added" : request.newtopic, "description": request.description}
    return {'message': 'topic extists'}
    
#part 3: endpoint to store emails
@app.post('/api/v1/emails/store')
def store_email(entry: EmailEntry):
    with open("/home/ec2-user/environment/lab2_factories/data/emails.json", 'r') as f:
        data=json.load(f)
    
    data.append(entry.model_dump())
    
    with open("/home/ec2-user/environment/lab2_factories/data/emails.json", "w") as f:
        json.dump(data,f)
    
    return{'message':'email stored'}
    
    
@app.get("/health")
def health_check():
     return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)