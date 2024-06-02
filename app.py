from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://tbpodbc'


db = SQLAlchemy(app)

# Define label dictionary mapping class labels to indices
label_dict = {
    'Vasculitis': 0,
    'Poison_Ivy_Dermatitis': 1,
    'Tinea_Ringworm_Candidiasis': 2,
    'Psoriasis_Lichen_Planus': 3,
    'Urticaria_Hives': 4,
    'Warts_Molluscum': 5,
    'Melanoma_Skin_Cancer': 6,
    'Eczema': 7,
    'Hair_Loss_Alopecia': 8,
    'Lupus': 9,
    'Bullous_Disease': 10,
    'Cellulitis_Impetigo': 11,
    'Nail_Fungus': 12,
    'Actinic_Keratosis_Basal_Cell_Carcinoma': 13,
    'Atopic_Dermatitis': 14,
    'Acne_and_Rosacea': 15
}


class Feedback(db.Model):
    __tablename__ = 'feedback'
    feedback = db.Column(db.String(255), primary_key=True)

    def __init__(self, feedback):
        self.feedback = feedback

def preprocess_image(image):
    # Resize image to (100, 100) and normalize pixel values
    img = image.resize((100, 100))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/choose_image')
def choose_image():
    return render_template('index2.html')

@app.route('/feedback')  # Define route for feedback page
def feedback():
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_text = request.form.get('feedback')

    # Create a new feedback object
    new_feedback = Feedback(feedback=feedback_text)

    # Add the new feedback to the database session
    db.session.add(new_feedback)
    db.session.commit()
    # Redirect to the feedbacks page
    return redirect(url_for('home'))
    

@app.route('/feedbacks')  # Define route for feedbacks page
def feedbacks():
    feedback_data = []
    with open('feedbacks.txt', 'r') as f:
        feedback_data = f.readlines()
    return render_template('feedbacks.html', feedback_data=feedback_data)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    image = Image.open(file)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = list(label_dict.keys())[predicted_class_index]

    return jsonify({'class': predicted_class, 'probability': float(prediction[0][predicted_class_index])})

@app.route('/cure/<predicted_disease>')
def get_cure(predicted_disease):
    cures = {
        'Vasculitis': 'Treatments may include corticosteroids, immunosuppressive drugs, and medications to control inflammation. Precautions include avoiding triggers such as certain medications or infections.',
        'Poison_Ivy_Dermatitis': 'Calamine lotion, antihistamines, and corticosteroid creams can help relieve symptoms. Precautions include wearing protective clothing when outdoors and washing skin immediately after contact with potential irritants.',
        'Tinea_Ringworm_Candidiasis': 'Antifungal creams or oral medications are commonly used. Keeping the affected area clean and dry can also help prevent recurrence.',
        'Psoriasis_Lichen_Planus': 'Treatments vary but may include topical corticosteroids, phototherapy, and systemic medications. Precautions include avoiding triggers like stress, injury to the skin, and certain medications.',
        'Urticaria_Hives': 'Antihistamines can help relieve itching and swelling. Avoiding known triggers such as certain foods, medications, or environmental factors is important.',
        'Warts_Molluscum': 'Treatment options include topical medications, cryotherapy, or surgical removal. Precautions include avoiding close contact with infected individuals and keeping skin clean and dry.',
        'Melanoma_Skin_Cancer': 'Regular skin checks with a dermatologist are crucial for early detection. Sun protection, including wearing sunscreen and protective clothing, is essential.',
        'Eczema': 'Moisturizers, corticosteroids, and immunomodulators are commonly used treatments. Avoiding triggers like certain fabrics, harsh soaps, and extreme temperatures can help manage symptoms.',
        'Hair_Loss_Alopecia': 'Treatments vary depending on the underlying cause but may include medications, topical treatments, and procedures like hair transplants. Precautions include maintaining a healthy diet and managing stress.',
        'Lupus': 'Treatment focuses on managing symptoms and may include medications to suppress the immune system. Sun protection and avoiding stress are important for managing symptoms.',
        'Bullous_Disease': 'Treatment depends on the type and severity of the condition but may include medications to reduce inflammation and prevent blister formation. Avoiding known triggers is important for managing symptoms.',
        'Cellulitis_Impetigo': 'Antibiotics are the mainstay of treatment. Keeping the affected area clean and avoiding sharing personal items can help prevent spread.',
        'Nail_Fungus': 'Antifungal medications, either topical or oral, are commonly used. Keeping nails clean and dry can help prevent recurrence.',
        'Actinic_Keratosis_Basal_Cell_Carcinoma': 'Treatment options include cryotherapy, topical medications, surgery, and photodynamic therapy. Regular skin checks and sun protection are important for prevention.',
        'Atopic_Dermatitis': 'Moisturizers, corticosteroids, and immunomodulators are commonly used treatments. Avoiding triggers like certain fabrics, harsh soaps, and extreme temperatures can help manage symptoms.',
        'Acne_and_Rosacea': 'Treatments may include topical medications, oral antibiotics, or isotretinoin. Avoiding triggers like certain foods, stress, and harsh skincare products can help manage symptoms.'
    }
    return jsonify({'cure': cures.get(predicted_disease, 'No cure found.')})

if __name__ == '__main__':
    app.run(debug=True)
