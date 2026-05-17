#  CUSTOMER REVIEW ANALYSIS ASSISTANT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# ── APP SETUP ────────────────────────────────────────────────
app = FastAPI(
    title="Customer Review Analysis Assistant",
    description="Few-shot LLM analysis: Sentiment + Reason + Support Reply",
    version="1.0.0"
)

# ── LOAD MODEL ONCE AT STARTUP ───────────────────────────────
print("⏳ Loading TinyLlama model...")
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto"
)
print("✅ Model ready!")


# ── JSON INPUT SCHEMA ────────────────────────────────────────
# What the user must send in their POST request
class ReviewRequest(BaseModel):
    review: str                     # required — the customer's review text
    product: str = "product"        # optional — product name (default: "product")


# ── JSON OUTPUT SCHEMA ───────────────────────────────────────
# What we send back — always the same 3 fields
class ReviewResponse(BaseModel):
    status: str
    review: str
    sentiment: str       # Positive / Negative / Neutral
    reason: str          # one-sentence explanation
    support_reply: str   # suggested reply to the customer


# ── FEW-SHOT EXAMPLES ────────────────────────────────────────
# PDF Concept: Few-Shot = show the model 2-5 labelled examples
# before giving it the real question. This "locks in" the format.
# Using Pakistani e-commerce context (Daraz) as taught in class.

FEW_SHOT_EXAMPLES = """
Example 1:
Review: "Bilkul bekaar product hai. Maine Daraz se order kiya tha, 3 din mein toot gaya. Waste of money!"
Sentiment: Negative
Reason: Customer received a defective product that broke within 3 days of purchase.
Support Reply: Dear Customer, we sincerely apologize for this experience. Please contact our support team with your order number and we will arrange a free replacement or full refund within 24 hours.

Example 2:
Review: "Excellent quality! Delivered in 2 days and packaging was perfect. Exactly as shown in pictures. Very happy with my purchase from this seller."
Sentiment: Positive
Reason: Customer is satisfied with fast delivery, accurate product description, and quality.
Support Reply: Dear Customer, thank you so much for your kind words! We are delighted to hear you had a great experience. We look forward to serving you again!

Example 3:
Review: "Product is okay. Nothing special. Delivery took longer than expected but item is usable. Average experience overall."
Sentiment: Neutral
Reason: Customer found the product acceptable but was disappointed by slow delivery.
Support Reply: Dear Customer, thank you for your honest feedback. We are working on improving our delivery times. We hope your next experience with us will exceed your expectations!
"""


# ── PROMPT BUILDER WITH f-STRING ─────────────────────────────
# PDF Concept: f-string injects the real review into the template
# PDF Concept: 6 Components — all 6 are present here
def build_prompt(review: str, product: str) -> str:

    # COMPONENT 1 — ROLE
    role = "You are a professional customer support sentiment analyst for an e-commerce platform."

    # COMPONENT 2 — INSTRUCTION
    instruction = "Analyze the customer review below and return exactly 3 fields: Sentiment, Reason, and Support Reply."

    # COMPONENT 3 — CONTEXT
    context = f"The customer reviewed a {product} purchased online. Your job is to help the support team respond quickly."

    # COMPONENT 4 — INPUT (injected via f-string)
    input_data = f"Review to analyze: \"{review}\""

    # COMPONENT 5 — OUTPUT FORMAT (locked by few-shot examples)
    output_format = "Respond in exactly this format:\nSentiment: [Positive/Negative/Neutral]\nReason: [one sentence]\nSupport Reply: [professional reply to customer]"

    # COMPONENT 6 — CONSTRAINTS
    constraints = "Sentiment must be exactly one of: Positive, Negative, or Neutral. Reason must be one sentence only. Support Reply must be polite and professional. Do NOT add any extra text."

    # ASSEMBLE THE FULL FEW-SHOT PROMPT
    full_prompt = f"""
{role}
{instruction}
{context}
{output_format}
{constraints}

Here are examples of how to analyze reviews:
{FEW_SHOT_EXAMPLES}
Now analyze this new review:
{input_data}
"""
    return full_prompt


# ── RESPONSE PARSER ──────────────────────────────────────────
# Extracts the 3 structured fields from the LLM's raw text output
def parse_response(raw_text: str) -> dict:
    sentiment = "Unknown"
    reason    = "Could not extract reason."
    reply     = "Thank you for your feedback."

    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("Sentiment:"):
            sentiment = line.replace("Sentiment:", "").strip()
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()
        elif line.startswith("Support Reply:"):
            reply = line.replace("Support Reply:", "").strip()

    # Normalize sentiment to exactly 3 allowed values
    if "positive" in sentiment.lower():
        sentiment = "Positive"
    elif "negative" in sentiment.lower():
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "reason": reason,
        "support_reply": reply
    }


# ── HEALTH CHECK ─────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "running", "message": "Review Analysis Assistant is live!"}


# ── MAIN ENDPOINT ─────────────────────────────────────────────
@app.post("/analyze", response_model=ReviewResponse)
def analyze_review(request: ReviewRequest):

    # ERROR HANDLING — empty review
    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review text cannot be empty."
        )

    # ERROR HANDLING — review too long
    if len(request.review) > 800:
        raise HTTPException(
            status_code=400,
            detail="Review is too long. Please keep it under 800 characters."
        )

    # BUILD THE FEW-SHOT PROMPT using f-string
    prompt = build_prompt(request.review, request.product)

    messages = [
        {
            "role": "system",
            # PDF Concept: ROLE component — sets the AI's expert identity
            "content": "You are a professional customer support analyst. Always respond in the exact format shown in the examples."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # LLM CALL — with explicit temperature, top_p, top_k
    try:
        output = pipe(
            messages,
            max_new_tokens=250,

            # PDF Concept: temperature=0.3 → LOW = factual, consistent
            # We want SAME format every time, not creative variation
            temperature=0.3,

            # PDF Concept: top_p=0.9 → consider top 90% likely tokens
            # Keeps answers focused but not completely robotic
            top_p=0.9,

            # PDF Concept: top_k=50 → only pick from top 50 candidate words
            # Prevents random/weird word choices in structured output
            top_k=50,

            do_sample=True,
            repetition_penalty=1.1   # avoids the model repeating itself
        )

        # Extract the assistant's reply
        raw_response = output[0]["generated_text"][-1]["content"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}"
        )

    # PARSE the 3 structured fields from the raw text
    parsed = parse_response(raw_response)

    # RETURN clean JSON response
    return ReviewResponse(
        status="success",
        review=request.review,
        sentiment=parsed["sentiment"],
        reason=parsed["reason"],
        support_reply=parsed["support_reply"]
    )


# ── RUN SERVER ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("review_assistant:app", host="0.0.0.0", port=8000, reload=False)