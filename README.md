# 🧠 Real-Time ASL to Text Converter with AI Correction

**An intelligent web application that translates American Sign Language (ASL) finger-spelling into text in real-time, leveraging a Generative AI (Gemini Pro) to correct, enhance, and provide context to the recognized text.**

<!-- Optional: Add a screenshot or GIF of the app in action here -->

---

## 📖 Problem Statement

To create a real-time, AI-powered web application that translates American Sign Language finger-spelling into text to bridge the communication gap, complete with intelligent correction and contextual suggestions.

## ✨ Key Features

*   **Real-Time Sign Recognition:** Utilizes a custom-trained Convolutional Neural Network (CNN) to classify ASL alphabet gestures from a live webcam feed.
*   **Intelligent Hand Detection:** The application automatically pauses when no hand is detected in the Region of Interest (ROI), resuming only when a hand is present.
*   **Dynamic Jumbling Logic:** Simulates real-world transcription errors by intentionally inserting random characters into the recognized sequence, creating a challenge for the correction model.
*   **AI-Powered Sentence Correction (Gemini Pro):** Integrates with the Google Gemini API to parse the jumbled, space-less text and reconstruct the user's intended sentence with proper grammar and capitalization.
*   **Contextual Enhancement:** The AI doesn't just correct the text; it also generates:
    *   **Suggestions:** Five alternative ways to phrase the corrected sentence.
    *   **Quick Replies:** Five common replies to the corrected sentence, making the interaction more dynamic.
*   **Fully Interactive Web Interface:** Built with Flask and Socket.IO for seamless, low-latency communication between the Python backend and the user's browser.

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, Flask-SocketIO |
| **Machine Learning** | PyTorch, OpenCV, NumPy, Pillow |
| **Generative AI** | Google Generative AI (Gemini Pro) |
| **Frontend** | HTML, CSS, JavaScript |

## ⚙️ Project Architecture (Client-Server Flow)

The application operates with a seamless client-server architecture:

1.  **Initialization:** The application is configured with a target sentence upon startup.
2.  **Web Client:** The user opens the web application. The frontend establishes a Socket.IO connection.
3.  **Video Streaming:** The frontend captures webcam frames and streams them to the Flask backend via Socket.IO.
4.  **Backend Processing:**
    *   The backend isolates the Region of Interest (ROI).
    *   It uses a skin-tone filter to robustly detect if a hand is present (pausing the game if not).
    *   If a hand is present, the frame is processed (grayscaled, thresholded) and fed into the PyTorch CNN model.
    *   The model's character prediction is sent back to the frontend.
5.  **Frontend Logic:** The recognized character and a random "jumble" character are appended to the sentence string, simulating real-world errors.
6.  **AI Correction Trigger:** When the user clicks **"Stop"**, the final jumbled sentence and the original intended phrase are sent to the backend.
7.  **Generative AI Call:** The backend constructs a detailed prompt and calls the **Google Gemini API**.
8.  **Final Display:** The AI's response (corrected sentence, suggestions, and replies) is parsed and displayed on the frontend.

---

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

*   Python 3.9+
*   A webcam
*   A [Google Generative AI API Key](https://ai.google.dev/gemini-api/docs/api-key)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Create a Virtual Environment
(It's highly recommended to use a virtual environment.)
code
Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies
Install all the required libraries using pip.
code
Bash
pip install -r requirements.txt
requirements.txt Content:
code
Code
Flask
Flask-SocketIO
torch
torchvision
Pillow
opencv-python
numpy
google-generativeai
4. Get the Pre-trained Model
Ensure the pre-trained model file, sign_language_model.pth, is present in the root directory of the project.
5. Set Up Your API Key
Open the app.py file.
Replace the placeholder 'YOUR_GOOGLE_GEMINI_API_KEY' with your actual API key.
code
Python
# --- V V V PASTE YOUR KEY HERE V V V ---
API_KEY = 'YOUR_GOOGLE_GEMINI_API_KEY' 
genai.configure(api_key=API_KEY)
# --- ^ ^ ^ PASTE YOUR KEY HERE ^ ^ ^ ---```
⚠️ **Important:** Do not commit your API key to a public repository.

## ▶️ How to Run

1.  Activate your virtual environment.
2.  Run the `app.py` script from your terminal:

```bash
python app.py
Once the server starts, open your web browser and navigate to: http://localhost:5000
🎮 How to Use
Start: Once the webpage loads, the "Start" button will be enabled. Click it and grant webcam permission.
Position Hand: Place your hand inside the glowing blue box (the Region of Interest). The application will detect your hand and the transcription will begin.
Sign: Perform the ASL signs for the letters of the target sentence. The application will build a "jumbled" version of the sentence in the display area.
Stop & Correct: When you are finished, click the "Stop" button.
View Results: The backend will send the jumbled text to the Gemini Pro API, and the frontend will display the clean, corrected sentence along with the AI-generated suggestions and replies.
🔮 Future Improvements
Expand the Vocabulary: Train the model to recognize full words or common phrases, not just individual letters.
Dynamic ROI: Implement a hand-tracking algorithm that allows the ROI to follow the user's hand around the screen.
Deployment: Deploy the application to a cloud service (like Heroku, AWS, or Google Cloud) to make it publicly accessible.
User Feedback: Add a feature for users to confirm if the AI's correction was accurate, which could be used to fine-tune the prompts.
License
This project is licensed under the MIT License. See the LICENSE file for details.
