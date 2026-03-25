import pickle
import numpy as np
import os

def predict_performance(sample_data):
    """
    sample_data: list or array of 8 features:
    [Study Hours per Week, Attendance Rate, Previous Grades, 
     Participation in Extracurricular Activities_Yes,
     Parent Education Level_Bachelor, Parent Education Level_Doctorate,
     Parent Education Level_High School, Parent Education Level_Master]
    """
    model_path = os.path.join('model', 'student_model.pkl')
    
    if not os.path.exists(model_path):
        return "Model file not found. Please run the notebook/model.ipynb first."
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    sample = np.array([sample_data])
    prediction = model.predict(sample)
    
    return "Likely to PASS" if prediction[0] == 1 else "Likely to FAIL"

if __name__ == "__main__":
    # Example: 10 Study Hours, 85% Attendance, 75 Previous Grades,
    # No Extracurriculars (0), Parent Education Level: High School (1 for High School)
    example_input = [10.0, 85.0, 75.0, 0, 0, 0, 1, 0]
    result = predict_performance(example_input)
    print(f"Prediction for input {example_input}: {result}")
