import time
import speech_recognition as sr
from groq import Groq
from PIL import ImageGrab, Image
from dotenv import load_dotenv
import google.generativeai as genai
import os
import base64
import asyncio
import edge_tts
import pygame
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
import json
import websockets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize pygame mixer
pygame.mixer.init()

# Load environment variables
load_dotenv()

# API Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")

groq_client = Groq(api_key=groq_api_key)
genai.configure(api_key=genai_api_key)

# RAG Setup
def get_embedding_function():
    embeddings = CohereEmbeddings(cohere_api_key="yLa4P1FNzncjNN90YZGTukQciYi2NtZs85WiavFY", model="embed-english-v3.0")
    return embeddings

def query_rag(query_text: str):
    rag_keywords = ["rag", "retrieved", "knowledge base", "stored knowledge", "what do you know about", "tell me about"]
    if any(keyword in query_text.lower() for keyword in rag_keywords):
        embedding_function = get_embedding_function()
        chroma_path = "chromaa"
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=1)
        valid_results = [(doc, score) for doc, score in results if doc.page_content and doc.page_content.strip()]
        if not valid_results:
            return "No relevant information found."
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in valid_results])
        return context_text
    return ""

# Model Configuration for Gemini
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# System message
sys_msg = (
    'You are a multi-modal AI voice assistant named Prometheus. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity. Also call me Master while replying.'
)

convo = [{'role': 'system', 'content': sys_msg}]

# Logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
log_messages = []

def log(message, title, style):
    console.print(Panel(Markdown(f"**{message}**"), border_style=style, expand=False, title=title))
    log_messages.append(f"[{title}] {message}")

def save_log():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}.txt"
    with open(filename, "w") as f:
        for message in log_messages:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(f"Log saved to {filename}")

# TTS with Edge TTS
async def speak(response, voice='en-US-MichelleNeural', output_file='temp_output.mp3'):
    log(f"Generating speech for: {response}", title="TTS", style="bold cyan")
    try:
        communicate = edge_tts.Communicate(response, voice)
        await communicate.save(output_file)
        log(f"Audio file saved: {output_file}", title="TTS", style="bold cyan")
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        log("Playing audio...", title="TTS", style="bold cyan")
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        log("Audio playback finished", title="TTS", style="bold cyan")
    except Exception as e:
        log(f"Error in speak: {str(e)}", title="ERROR", style="bold red")
    finally:
        try:
            pygame.mixer.music.unload()
            if os.path.exists(output_file):
                os.remove(output_file)
                log(f"Cleaned up: {output_file}", title="TTS", style="bold cyan")
        except Exception as e:
            log(f"Cleanup error: {str(e)}", title="ERROR", style="bold red")

# Groq Prompt Processing
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(
        messages=convo,
        model='llama3-70b-8192'
    )
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

# Function Call Logic
def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model named AVA. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["take_screenshot", "capture_webcam", "None"]. '
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )
    convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    chat_completion = groq_client.chat.completions.create(
        messages=convo,
        model='llama3-70b-8192'
    )
    response = chat_completion.choices[0].message
    return response.content

# Screenshot Functionality
def take_screenshot():
    log("Taking screenshot...", title="ACTION", style="bold blue")
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return path

# Vision Prompt with Gemini
def vision_prompt(prompt, photo_path):
    log("Generating vision prompt...", title="ACTION", style="bold blue")
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI '
        'assistant to the user. Instead take the user prompt input and try to extract all meaning '
        'from the photo relevant to the user prompt. Then generate as much objective data about '
        'the image for the AI assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    log("Vision prompt generated.", title="ACTION", style="bold blue")
    return response.text

# WebSocket Handler
async def handle_connection(websocket):
    log("WebSocket connection established.", title="ACTION", style="bold blue")
    try:
        async for message in websocket:
            log(f"Received message: {message}", title="WEBSOCKET", style="bold yellow")
            data = json.loads(message)
            user_input = data.get("prompt")
            is_text_input = data.get("isTextInput", False)
            webcam_image = data.get("webcamImage", None)
            
            if user_input and user_input.lower() != "could not understand audio":
                log(f'USER INPUT: {user_input}', title="USER", style="bold green")
                
                # Process RAG
                rag_text = query_rag(user_input)
                rag = str(rag_text)
                
                # Function call and vision context
                call = function_call(user_input)
                visual_context = None
                image_data = None
                
                if webcam_image:  # Webcam image from frontend
                    log("Processing real-time webcam image...", title="ACTION", style="bold blue")
                    photo_path = 'webcam_from_frontend.jpg'
                    with open(photo_path, 'wb') as f:
                        f.write(base64.b64decode(webcam_image))
                    visual_context = vision_prompt(prompt=user_input + rag, photo_path=photo_path)
                    image_data = webcam_image  # Send back the same image
                    
                elif 'take_screenshot' in call:
                    photo_path = take_screenshot()
                    visual_context = vision_prompt(prompt=user_input + rag, photo_path=photo_path)
                    image_data = base64.b64encode(open(photo_path, 'rb').read()).decode('utf-8')
                    
                elif 'capture_webcam' in call:
                    log("Webcam capture requested; using frontend image instead.", title="INFO", style="bold yellow")
                    visual_context = "Please use the webcam button in the UI for real-time capture."

                # Get response from Groq
                response = groq_prompt(prompt=user_input + rag, img_context=visual_context)
                
                # Log and send response
                log(f'ASSISTANT OUTPUT: {response}', title="ASSISTANT", style="bold magenta")
                await websocket.send(json.dumps({
                    "role": "assistant",
                    "content": response,
                    "image": image_data
                }))
                log("Response sent to client.", title="WEBSOCKET", style="bold yellow")
                
                # Speak response only if not text input
                if not is_text_input:
                    log("Attempting to speak response...", title="TTS", style="bold cyan")
                    await speak(response)
                    log("Speak function completed.", title="TTS", style="bold cyan")
                else:
                    log("Text input detected, skipping voice output.", title="TTS", style="bold cyan")
                
                # Save to report
                with open('report.txt', 'a') as file:
                    file.write(f"User: {user_input}\n")
                    file.write(f"AI: {response}\n")
                    file.write("-" * 40 + "\n")
    except websockets.ConnectionClosed:
        log("WebSocket connection closed by client.", title="WEBSOCKET", style="bold yellow")
    except Exception as e:
        log(f"Error in WebSocket handler: {str(e)}", title="ERROR", style="bold red")
    finally:
        log("WebSocket connection terminated.", title="ACTION", style="bold blue")
        save_log()

async def main():
    server = await websockets.serve(handle_connection, "localhost", 8765)
    log("WebSocket server started on ws://localhost:8765", title="ACTION", style="bold blue")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())