from fastapi import FastAPI, Request, Form, BackgroundTasks
import httpx
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

def rate_limit_custom_handler(request: Request, exc: RateLimitExceeded):
    return HTMLResponse(
        content="Too many request, vũ trụ đang load vui lòng thử lại sau",
        status_code=429
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_custom_handler)

# Mount static files
app.mount("/static", StaticFiles(directory="web"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web/templates")

# Load Hugging Face model (Zero-Shot Classification)
# Using a multilingual model that works well with Vietnamese
try:
    # mDeBERTa-v3-base-mnli-xnli is excellent for multilingual zero-shot classification
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    classifier = None


import json
import random
from datetime import datetime

# Load answers from JSON file
try:
    with open("answers.json", "r", encoding="utf-8") as f:
        ANSWERS_DATA = json.load(f)
except Exception as e:
    print(f"Error loading answers.json: {e}")
    ANSWERS_DATA = []

# Map labels to Status Codes
LABEL_MAPPING = {
    "Sự nghiệp": "S0",
    "Công việc": "S0",
    "Tình duyên": "S1",
    "Tình yêu": "S1",
    "Gia đạo": "S1", 
    "Sức khỏe": "S2",
    "Tài lộc": "S3",
    "Tiền bạc": "S3",
    "Tổng quát": "S4",
    "Khác": "S4"
}

# Candidate labels for the model
CANDIDATE_LABELS = ["Sự nghiệp", "Tình duyên", "Sức khỏe", "Tài lộc", "Tổng quát"]
# Webhook URL
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

async def send_webhook(name: str, age: int, question: str):
    if not WEBHOOK_URL:
        return
    
    payload = {
        "text": f"New Fortune Request:\nName: {name}\nAge: {age}\nQuestion: {question}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"Error sending webhook: {e}")


def get_fortune(name: str, age: int, question: str) -> str:
    """
    Generate fortune by:
    1. Using Zero-Shot Classification to categorize the question (multi-label).
    2. Selecting random answers from the corresponding topics in answers.json.
    3. Formatting the response.
    """
    
    # 1. Determine current year and birth year
    current_year = datetime.now().year
    birth_year = current_year - age
    
    # 2. Classify question using AI
    statuses = set()
    topic_labels = []

    if classifier:
        try:
            # Zero-shot classification with multi-label support
            result = classifier(question, CANDIDATE_LABELS, multi_label=True)
            
            # Filter labels with score > 0.6
            # result['labels'] and result['scores'] are sorted by score descending
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.6:
                    print(f"AI classified '{name}' (age {age}) asked '{question}' -> Found '{label}' with score {score:.4f}")
                    if label in LABEL_MAPPING:
                        statuses.add(LABEL_MAPPING[label])
                        topic_labels.append(label)
            
        except Exception as e:
            print(f"Error classifying question: {e}")
    
    # Fallback if no specific topics found
    if not statuses:
        statuses.add("S4") # General
        topic_labels.append("Tổng quát")
    
    # 3. Select answers based on statuses
    selected_answers = []
    
    for status in statuses:
        # Find the category data
        category_data = next((item for item in ANSWERS_DATA if item["status"] == status), None)
        
        if category_data and "answers" in category_data:
            selected_answers.append(random.choice(category_data["answers"]))
            
    if not selected_answers:
        selected_answers.append("Vũ trụ đang gửi tín hiệu nhiễu, nhưng hãy tin rằng mọi điều tốt đẹp sẽ đến.")

    # 4. Format response
    # Join multiple answers if any
    combined_answer = " ".join(selected_answers)
    final_response = f"Gia chủ {name}, Sinh năm {birth_year}. {combined_answer}"
    
    return final_response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page 1: Display input form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
@limiter.limit("5/hour")
async def predict(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    question: str = Form(...)
):
    """Handle form submission and redirect to loading page"""
    # Validate inputs
    if not name or not question:
        return RedirectResponse(url="/", status_code=303)
    
    if age < 1 or age > 120:
        return RedirectResponse(url="/", status_code=303)
    
    # Redirect to loading page with data
    return templates.TemplateResponse(
        "loading.html",
        {
            "request": Request(scope={"type": "http"}),
            "name": name,
            "age": age,
            "question": question
        }
    )


@app.post("/result", response_class=HTMLResponse)
@limiter.limit("5/hour")
async def result(
    request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    age: int = Form(...),
    question: str = Form(...)
):
    """Page 3: Generate and display fortune-telling result"""
    # Generate fortune using AI
    fortune = get_fortune(name, age, question)
    
    # Send webhook
    background_tasks.add_task(send_webhook, name, age, question)
    
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "name": name,
            "age": age,
            "question": question,
            "fortune": fortune
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
