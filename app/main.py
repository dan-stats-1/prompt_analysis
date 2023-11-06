from fastapi import FastAPI
from pydantic import BaseModel

from model import JailbreakDetector

app = FastAPI()

detector = JailbreakDetector()
detector.load_model("trained_model.pickle")

class Prompt(BaseModel):
    prompt: str

@app.post("/classify")
async def score_prompt(prompt: Prompt):
    action = detector.classify(prompt.prompt)
    return {"action": action}
