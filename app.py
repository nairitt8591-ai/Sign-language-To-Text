import os
from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import base64
import google.generativeai as genai

# --- API Key Configuration (Secure method recommended, but using hardcoded as requested) ---
# NOTE: Using os.environ.get for the SECRET_KEY is a good practice, even if you keep the API_KEY hardcoded.
API_KEY = 'AIzaSyB7aTZ9-eEZNBMUH2Xab2zbYWDQj2JBgCA' 
genai.configure(api_key=API_KEY)

# --- FIX: Define a default value for the phrase (remove the input() call) ---
# Since you cannot get input from the terminal on Render,
# we start with a default phrase. The client must provide the real phrase.
# You can set this to whatever you want, or just leave it empty.
# If your client (index.html/JS) provides the phrase, this variable is less critical.
intended_phrase_from_terminal = "Hello world I am happy"
print(f"Server will start with default sentence: '{intended_phrase_from_terminal}'")


# --- Flask App and WebSocket Setup ---
app = Flask(__name__)
# Use an Environment Variable for a real secret key on Render!
# Render will automatically set a PORT environment variable.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key') 
socketio = SocketIO(app)

# --- 1. MODEL ARCHITECTURE (No changes) ---
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1); self.pool = nn.MaxPool2d(kernel_size=2, stride=2); self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1); self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1); self.flatten = nn.Flatten(); self.fc1 = nn.Linear(128 * 16 * 16, 256); self.dropout = nn.Dropout(0.5); self.fc2 = nn.Linear(256, num_classes); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))); x = self.pool(self.relu(self.conv2(x))); x = self.pool(self.relu(self.conv3(x))); x = self.flatten(x); x = self.relu(self.fc1(x)); x = self.dropout(x); x = self.fc2(x)
        return x

# --- 2. CONFIGURATION AND LOADING ---
MODEL_PATH = 'sign_language_model.pth'
CLASS_NAMES = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
NUM_CLASSES = len(CLASS_NAMES); IMG_SIZE = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Try/Except block for model loading (important for deployment stability) ---
try:
    model = SignLanguageCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    print(f"ERROR: Could not load model from {MODEL_PATH}. Check file path/size. Error: {e}")
    # You might want to exit or handle this gracefully. For now, continue...

data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize(IMG_SIZE), transforms.ToTensor()])
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

# --- 3. Flask Routes and WebSocket Events ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Send the default sentence to the client when they connect ---
# The client must be updated to let the user input the desired phrase.
@socketio.on('connect')
def handle_connect():
    print(f"Client connected. Sending default sentence: '{intended_phrase_from_terminal}'")
    socketio.emit('set_sentence', {'sentence': intended_phrase_from_terminal})


@socketio.on('image')
def handle_image(data_url):
    img_data = base64.b64decode(data_url.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = frame.shape
    x1 = 0; y1 = 0; x2 = int(w * 0.5); y2 = int(h * 0.5)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_present = False
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 3000:
            hand_present = True
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, processed_img = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(processed_img)
    input_tensor = data_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    predicted_char = CLASS_NAMES[predicted_idx.item()]
    display_img = cv2.flip(processed_img, 1)
    processed_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.jpg', processed_bgr)
    processed_b64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('response', {'char': predicted_char, 'processed_image': processed_b64, 'hand_present': hand_present})

@socketio.on('correct_sentence')
def handle_correction(data):
    jumbled_sentence = data.get('sentence')
    intended_phrase = data.get('intended_phrase')
    if not jumbled_sentence or not intended_phrase:
        return
        
    # --- THIS PROMPT IS THE ONLY THING THAT NEEDS TO CHANGE ---
    prompt = f"""A user was trying to communicate the phrase "{intended_phrase.lower()}".

Perform the following three tasks based on that phrase. Use the headers exactly as written (CORRECTED PHRASE:, SUGGESTIONS:, REPLIES:).

CORRECTED PHRASE:
Take the user's phrase "{intended_phrase.lower()}" and rewrite it as a natural, grammatically correct sentence with proper spacing and capitalization.

SUGGESTIONS:
1. Provide five friendly, alternative ways to say the corrected phrase.

REPLIES:
1. Provide five common and friendly quick replies to the corrected phrase.
"""
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        # The rest of the function remains the same
        corrected_text = text.split('SUGGESTIONS:')[0].replace('CORRECTED PHRASE:', '').strip()
        suggestions_text = text.split('SUGGESTIONS:')[1].split('REPLIES:')[0]
        suggestions = [line.strip() for line in suggestions_text.strip().split('\n') if line.strip()]
        replies_text = text.split('REPLIES:')[1]
        replies = [line.strip() for line in replies_text.strip().split('\n') if line.strip()]
        socketio.emit('final_sentence', {
            'sentence': corrected_text, 
            'suggestions': suggestions,
            'replies': replies
        })
    except Exception as e:
        print(f"Gemini API error: {e}")
        socketio.emit('final_sentence', {
            'sentence': "Error generating response.", 
            'suggestions': [], 
            'replies': []
        })

# --- 4. Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # Use the PORT environment variable provided by Render when deploying
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)