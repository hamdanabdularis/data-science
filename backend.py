import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app (this creates the 'app' variable Uvicorn is looking for!)
app = FastAPI(title="Student Depression Prediction API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("student_depression_model.pkl")


# Define expected input structure
class StudentInput(BaseModel):
  gender: int  # 1 for Male, 0 for Female
  age: int
  academic_pressure: float
  cgpa: float
  study_satisfaction: float
  sleep_duration: int
  dietary_habits: int
  work_study_hours: float
  financial_stress: int


@app.post("/predict")
def predict_depression(data: StudentInput):
  # Construct input array for the model
  features = [
      data.gender,
      data.age,
      0,
      0,
      data.academic_pressure,
      0,
      data.cgpa,
      data.study_satisfaction,
      0,
      data.sleep_duration,
      data.dietary_habits,
      0,
      0,
      data.work_study_hours,
      data.financial_stress,
      0,
  ]

  prediction = model.predict([features])[0]
  probability = (
      model.predict_proba([features])[0][1]
      if hasattr(model, "predict_proba")
      else 0.0
  )

  return {
      "prediction": int(prediction),
      "risk_status": "High Risk" if prediction == 1 else "Low Risk",
      "probability": float(probability),
  }