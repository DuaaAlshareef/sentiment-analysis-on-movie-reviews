from fastapi import FastAPI
from pydantic import BaseModel
from model.model import load_model, predict_sentiment
from data.preprocessing import preprocess_text

# Initialize FastAPI
app = FastAPI()

# Load the model and tokenizer
model, device = load_model()

# Define the input data structure using Pydantic
class ReviewInput(BaseModel):
    review: str

input_tensor = preprocess_text("bAD MOVIE NO LIKE")
        
# Make the prediction
sentiment_label, probability = predict_sentiment(model, device, input_tensor)

# Define the prediction endpoint
@app.post("/predict")
def predict(data: ReviewInput):
    try:
        # Preprocess the review text
        input_tensor = preprocess_text(data.review)
        
        # Make the prediction
        sentiment_label, probability = predict_sentiment(model, device, input_tensor)
        
        # Return the result
        return {"review": data.review, "sentiment": sentiment_label, "Model Confidence Score": probability}
    
    except Exception as e:
        return {"error": str(e), "message": "An error occurred during the prediction process."}


@app.get("/health")
def health_check():
    return {"status": "API is running", "model_loaded": model is not None}
