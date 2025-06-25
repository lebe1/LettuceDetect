import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import AsyncGenerator

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_VALID_CHAR_REGEX = re.compile(r"^[\x20-\x7E\n\r\t]+$")

class PromptRequest(BaseModel):
    prompt: str
    regex: str = ""

def extract_digits(text: str) -> set:
    return set(re.findall(r"\d", text))

def validate_token(token: str, allowed_digits: set) -> bool:
    found_digits = set(re.findall(r"\d", token))
    return not (found_digits - allowed_digits)

def is_printable(text: str, regex: re.Pattern) -> bool:
    return regex.fullmatch(text) is not None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/stream")
async def stream(prompt_request: PromptRequest) -> StreamingResponse:
    prompt = prompt_request.prompt
    regex_pattern = prompt_request.regex.strip() or r"^[\x20-\x7E\n\r\t]+$"

    try:
        valid_char_regex = re.compile(regex_pattern)
    except re.error:
        valid_char_regex = DEFAULT_VALID_CHAR_REGEX

    allowed_digits = extract_digits(prompt)

    async def token_generator() -> AsyncGenerator[str, None]:
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                output_so_far = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = httpx.Response(200, content=line).json()
                        token = data.get("response", "")
                        output_so_far += token

                        if not is_printable(output_so_far, valid_char_regex):
                            yield "\n[!] Invalid characters detected. Stopping generation."
                            break

                        if not validate_token(token, allowed_digits):
                            yield f"\n[!] Invalid digit(s) in token: '{token}'. Stopping."
                            break

                        yield token
                    except Exception as e:
                        yield f"\n[!] Error: {str(e)}"
                        break

    return StreamingResponse(token_generator(), media_type="text/plain")
