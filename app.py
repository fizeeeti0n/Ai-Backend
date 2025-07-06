from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import PyPDF2
import io
import requests
import os
from dotenv import load_dotenv
import base64
from docx import Document # Import for .docx files
from PIL import Image # For image processing (used by pytesseract)
import pytesseract # For OCR on images (if needed)

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__)

# --- Configure CORS more explicitly ---
# In production, specify exact origins. For troubleshooting, you can keep '*'
# But it's highly recommended to change to specific origins for security.
FRONTEND_ORIGIN = "https://uni-ai-2q9g.onrender.com"
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN, "methods": ["GET", "POST", "OPTIONS"], "headers": ["Content-Type", "Authorization"]}})

# If the above CORS line doesn't fix it, uncomment the @app.after_request block below
# and redeploy. This acts as a fallback or a more direct way to set headers.
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', FRONTEND_ORIGIN)
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true') # If you're using cookies/auth
#     return response


# --- Retrieve API Key ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("\n" * 3) # Add some spacing for visibility
    print("-" * 70)
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable NOT SET.")
    print("Please ensure your .env file is in the same directory as app.py")
    print("and contains: GEMINI_API_KEY='YOUR_ACTUAL_GEMINI_API_KEY_HERE'")
    print("Alternatively, hardcode it in app.py for testing (NOT RECOMMENDED FOR PRODUCTION).")
    print("-" * 70)
    print("\n" * 3)
    # Uncomment the line below for quick local testing if you don't want to use .env
    # API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY_HERE" # <--- Replace with your real API key for testing

# --- Define Gemini API URLs ---
GEMINI_TEXT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_VISION_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

# --- Configure Tesseract CMD Path (if not in system PATH) ---
# For Windows, it might look like:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/macOS, if it's not in PATH, you'd specify the full path to 'tesseract' executable.
# Example for a common Linux path:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

@app.route('/upload', methods=['POST', 'OPTIONS'])
@cross_origin() # This decorator is redundant if CORS is configured globally, but harmless
def upload_file():
    """
    Handles file uploads (PDFs and DOCX for text extraction, images for text extraction or acknowledgment).
    Supports CORS preflight requests (OPTIONS).
    """
    if request.method == 'OPTIONS':
        # Preflight request, just return 200 OK
        return '', 200

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    filename_lower = file.filename.lower()
    mimetype = file.mimetype

    extracted_text = ""
    file_type_processed = "unknown" # To send back what type was processed

    try:
        # --- Handle PDF Files ---
        if mimetype == 'application/pdf':
            pdf_file = io.BytesIO(file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted_text_from_page = page.extract_text()
                if extracted_text_from_page:
                    extracted_text += extracted_text_from_page + "\n"
            
            if not extracted_text.strip():
                return jsonify({'success': False, 'message': 'Could not extract text from PDF. It might be image-based or empty.'}), 400
            file_type_processed = 'pdf'

        # --- Handle DOCX Files ---
        elif mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            docx_file = io.BytesIO(file.read())
            document = Document(docx_file)
            for paragraph in document.paragraphs:
                extracted_text += paragraph.text + "\n"
            
            if not extracted_text.strip():
                return jsonify({'success': False, 'message': 'Could not extract text from DOCX. It might be empty.'}), 400
            file_type_processed = 'docx'

        # --- Handle Image Files (Optional: Add OCR here if needed) ---
        elif mimetype.startswith('image/'):
            try:
                img = Image.open(io.BytesIO(file.read()))
                extracted_text = pytesseract.image_to_string(img)
                if not extracted_text.strip():
                    return jsonify({'success': False, 'message': 'Could not extract text from image via OCR. Text might be too blurry or not present.'}), 400
                message = 'Image processed successfully via OCR!'
                file_type_processed = 'image_ocr'
            except pytesseract.TesseractNotFoundError:
                print("Tesseract OCR engine not found. Please install it.")
                return jsonify({'success': False, 'message': 'Tesseract OCR engine not found. Cannot process image for text.'}), 500
            except Exception as e:
                print(f"Error during image OCR: {e}")
                return jsonify({'success': False, 'message': f'Error during image OCR: {str(e)}'}), 500
            
        else:
            return jsonify({'success': False, 'message': f'Unsupported file type: {file.filename}. Please upload a PDF, DOCX, or an image.'}), 400

        # Return success response with extracted text if any, and file type
        return jsonify({
            'success': True,
            'message': f'{file_type_processed.upper()} processed successfully!',
            'extractedText': extracted_text,
            'fileType': file_type_processed
        }), 200

    except Exception as e:
        print(f"Error during file processing: {e}")
        return jsonify({'success': False, 'message': f'Server error during file processing: {str(e)}
