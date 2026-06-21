from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Greetings(BaseModel):
    name: str
    excited: bool

@app.get('/')
async def read_root():
    return {"message": "Hello from FastAPI"}

@app.get('/hello/{name}')
async def say_hello(name:str):
    return {"greeting": f"Hello {name}!"}

@app.post('/greet')
async def post_greeting(payload: Greetings):
    punctuation="!!!" if payload.excited else "."
    return {"greeting":f"Hello {payload.name}{punctuation}"}

