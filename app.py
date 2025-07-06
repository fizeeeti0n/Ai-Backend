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
# Ensure CORS is configured properly. For development, allow all origins.
# In production, specify exact origins: origins=["http://127.0.0.1:5500", "https://your-frontend-domain.com"]
CORS(app, resources={r"/*": {"origins": "*"}})

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
@cross_origin()
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
            # Current logic: Just acknowledge receipt of image. Frontend handles base64 for multimodal.
            # If you want to extract text from images (e.g., scanned documents), enable OCR here.
            
            # Option 1: Acknowledge image, no text extraction on backend (your current logic)
            # This is fine if your AI model handles image content via Base64 from the frontend.
            # extracted_text = "" # No text extracted on backend for this image
            # message = 'Image received successfully (not processed for text on server).'
            # file_type_processed = 'image'

            # Option 2: Perform OCR on the image to extract text
            # This requires Tesseract-OCR engine and pytesseract library
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
        return jsonify({'success': False, 'message': f'Server error during file processing: {str(e)}'}), 500

@app.route('/summarize', methods=['POST', 'OPTIONS'])
@cross_origin()
def summarize():
    if request.method == 'OPTIONS': return '', 200
    if not API_KEY: return jsonify({'summary': 'API key not configured.'}), 500
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        if not documents: return jsonify({'summary': 'No text documents provided to summarize.'}), 400
        full_text = "\n\n".join(documents)
        
        # Ensure the prompt doesn't exceed model limits if full_text is very long
        # Consider truncating or sending in chunks if this becomes an issue
        prompt = f"Please summarize the following text:\n\n{full_text}"
        
        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history, "generationConfig": {"maxOutputTokens": 500}}
        
        response = requests.post(f"{GEMINI_TEXT_API_URL}?key={API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        summary = result['candidates'][0]['content']['parts'][0]['text'] if result.get('candidates') else "Could not generate summary."
        return jsonify({'summary': summary}), 200
    except requests.exceptions.RequestException as req_e: 
        print(f"API Request Error for summarize: {req_e}")
        return jsonify({'summary': f'API Error: {str(req_e)}. Check your API key and permissions.'}), 500
    except Exception as e: 
        print(f"General Error for summarize: {e}")
        return jsonify({'summary': f'Error: {str(e)}'}), 500

@app.route('/ask', methods=['POST', 'OPTIONS'])
@cross_origin()
def ask():
    if request.method == 'OPTIONS': return '', 200
    if not API_KEY: return jsonify({'answer': 'API key not configured.'}), 500
    try:
        data = request.get_json()
        question = data.get('question', '')
        documents = data.get('documents', [])
        images_b64 = data.get('images', [])
        department = data.get('department', '')
        chat_history = data.get('chatHistory', [])

        if not question and not images_b64 and not documents: # Ensure at least one input for AI
            return jsonify({'answer': 'No question, images, or documents provided for AI to process.'}), 400

        contents_parts = []

        context_text = "\n\n".join(documents)
        if department:
            context_text = f"The question is related to the '{department}' department.\n\n" + context_text

        # Append context/question for the current turn
        if question and context_text:
            contents_parts.append({"text": f"Using the following context, answer the question:\n\nContext:\n{context_text}\n\nQuestion: {question}"})
        elif question: # Only question, no docs
            contents_parts.append({"text": question})
        elif context_text: # Only docs, no specific question
             contents_parts.append({"text": f"Context for AI to analyze: {context_text}"})
        
        for img_data_url in images_b64:
            if ',' in img_data_url:
                try:
                    mime_type = img_data_url.split(';')[0].split(':')[1]
                    base64_data = img_data_url.split(',')[1]
                    contents_parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_data
                        }
                    })
                except Exception as e:
                    print(f"Error processing image data URL: {e} for {img_data_url[:50]}...")
                    continue
            else:
                print(f"Warning: Malformed image data URL received: {img_data_url[:50]}...")
                continue

        final_contents = []
        # Reconstruct chat history in the correct format for Gemini API
        for chat_entry in chat_history:
            # Ensure chat_entry is a dictionary and has 'role' and 'parts' or 'text'
            if isinstance(chat_entry, dict):
                if 'parts' in chat_entry and isinstance(chat_entry['parts'], list):
                    final_contents.append(chat_entry)
                elif 'text' in chat_entry: # Handle simpler history format if sent by frontend
                    final_contents.append({"role": chat_entry.get('role', 'user'), "parts": [{"text": chat_entry['text']}]})
        
        # Add the current user turn
        final_contents.append({"role": "user", "parts": contents_parts})

        payload = {
            "contents": final_contents,
            "generationConfig": {"maxOutputTokens": 800}
        }
        
        api_url = GEMINI_VISION_API_URL if images_b64 else GEMINI_TEXT_API_URL
        
        response = requests.post(f"{api_url}?key={API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        
        answer = "Sorry, I could not generate an answer."
        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            for part in result['candidates'][0]['content']['parts']:
                if 'text' in part:
                    answer = part['text']
                    break
        elif result.get('error'):
            answer = f"AI Error: {result['error'].get('message', 'Unknown API error')}. Code: {result['error'].get('code', 'N/A')}"
            print(f"Gemini API Error Response: {result['error']}")

        return jsonify({'answer': answer}), 200
    except requests.exceptions.RequestException as req_e: 
        print(f"API Request Error for ask: {req_e}")
        return jsonify({'answer': f'API Error: {str(req_e)}. Check your API key, model permissions, and request format.'}), 500
    except Exception as e: 
        print(f"General Error for ask: {e}")
        return jsonify({'answer': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) # Specify host for explicit binding
