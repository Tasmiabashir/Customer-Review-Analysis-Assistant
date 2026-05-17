# Customer Review Analysis Assistant

POST a customer review to `/analyze` and get **sentiment** (Positive/Negative/Neutral), a **reason**, and a **support reply** — powered by TinyLlama with few-shot examples (Daraz-style e-commerce).

**Stack:** Python · FastAPI · Uvicorn · Pydantic · Transformers · TinyLlama

**Run:** `pip install fastapi uvicorn transformers torch accelerate pydantic` then `python review_assistant.py` → open http://localhost:8000/docs
