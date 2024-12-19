from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1p1beta1 as speech
from google.api_core import exceptions as google_exceptions
from openai import OpenAI
import asyncio
import json
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from gtts import gTTS
import base64
import io
import re
from typing import Dict

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speech-to-text-denta.json"

# Initialize OpenAI client
client = OpenAI()

# Store active connections and conversation histories
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, list] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = []
        return self.active_connections[websocket]

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    def get_conversation_history(self, websocket: WebSocket):
        return self.active_connections.get(websocket, [])

    def update_conversation_history(self, websocket: WebSocket, history):
        self.active_connections[websocket] = history

manager = ConnectionManager()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_ordinal(n):
    """Convert number to ordinal string (1st, 2nd, 3rd, etc.)"""
    ordinal_numbers = {
        0: "first",
        1: "second",
        2: "third",
        3: "fourth",
        4: "fifth",
        5: "sixth",
        6: "seventh",
        7: "eighth",
        8: "ninth",
        9: "tenth"
    }
    return ordinal_numbers.get(n, str(n + 1) + "th")

def get_field_name(field):
    """Convert field path to natural name"""
    if 'dateTime' in field:
        if 'start' in field:
            return 'starting time'
        elif 'end' in field:
            return 'ending time'
        return 'time'
    
    if 'attendees' in field and 'email' in field:
        match = re.search(r'\[(\d+)\]', field)
        if match:
            index = int(match.group(1))
            return f"{get_ordinal(index)} attendee's email"
    
    # Simple field names
    simple_names = {
        'location': 'location',
        'description': 'description',
        'summary': 'summary'
    }
    
    for key, value in simple_names.items():
        if key in field:
            return value
    
    return field.split('.')[-1]

def check_null_values(data):
    """Check for NULL values and create a natural message"""
    missing_fields = []
    
    def check_dict(d, parent_key=''):
        for key, value in d.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                check_dict(value, current_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        check_dict(item, f"{current_key}[{i}]")
                    elif item == "NULL":
                        missing_fields.append(current_key)
            elif value == "NULL":
                missing_fields.append(current_key)
    
    check_dict(data)
    
    if not missing_fields:
        return None

    natural_fields = [get_field_name(field) for field in missing_fields]
    
    time_fields = [f for f in natural_fields if 'time' in f]
    other_fields = [f for f in natural_fields if 'time' not in f]
    
    message_parts = []
    
    if len(time_fields) == 2:
        message_parts.append('starting and ending time')
    elif time_fields:
        message_parts.extend(time_fields)
    
    message_parts.extend(other_fields)
    
    if message_parts:
        if len(message_parts) > 1:
            last_item = message_parts.pop()
            fields_text = ', '.join(message_parts) + ' and ' + last_item
        else:
            fields_text = message_parts[0]
        
        return f"Please provide {fields_text}"
    
    return None

def create_audio_response(text):
    """Create audio response using gTTS"""
    tts = gTTS(text=text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
    return audio_base64

async def process_with_llm(text: str, websocket: WebSocket):
    """Process the text with OpenAI and return response"""
    conversation_history = manager.get_conversation_history(websocket)
    
    system_prompt="""
    You are a JSON creator. When you will be given with a prompt you have to to extract the important/
    points from it and fill the following JSON format:
    ---
    {
        "summary": "Summary you will extract and if not given make one by yourself according to data",
        "location": "Location you will got in text",
        "description": "If user give the purpose or description, add here, if not create one by yourself",
        "colorId": "6",
        "start": {
            "dateTime": "The one user will given, make that in this format: 2024-08-15T09:00:00+05:30",
            "timeZone": "Asia/Karachi"
        },
        "end": {
            "dateTime": "The one user will given, make that in this format: 2024-08-15T09:40:00+05:30",
            "timeZone": "Asia/Karachi"
        },
        "recurrence": [
            "RRULE:FREQ=DAILY;COUNT=1"
        ],
        "attendees": [
            {
                "email": "example@example.com"
            }
        ]
    }
---
Your Final Response will be a JSON exactly the format above, You can say Fill the above JSON.
# NO PREAMBLE, ONLY VALID JASON #
If user do not provide the information which is need to be filled, Place a NULL at that position.
If user do not provide the end time of the meeting, add 30 minutes to the original time by yourself e.g. if user said at 5 pm, add end date as 5:30 pm.
You have history in memory, If user provide any remaining or information in another message, replace it with NULL or existing information and update the JSON.
Stick to the format above, nothing by yourself.
---
Here are few examples for you How to do that:
user_query: Create a meeting which is going to happen online at 5 thirty pm.
AI response:
    {
        "summary": "online meeting at 5:30 pm.",
        "location": "online",
        "description": "Meeting",
        "colorId": "6",
        "start": {
            "dateTime": "2024-12-17T09:00:00+05:30",
            "timeZone": "Asia/Karachi"
        },
        "end": {
            "dateTime": "2024-12-17T09:40:00+06:00",
            "timeZone": "Asia/Karachi"
        },
        "recurrence": [
            "RRULE:FREQ=DAILY;COUNT=1"
        ],
        "attendees": [
            {
                "email": "NULL"
            }
        ]
    }
"""
    
    if not any('system' in msg for msg in conversation_history):
        conversation_history.insert(0, {'role': 'system', 'content': system_prompt})
    
    conversation_history.append({'role': 'user', 'content': text})
    manager.update_conversation_history(websocket, conversation_history)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=conversation_history
        )
        response = completion.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': response})
        manager.update_conversation_history(websocket, conversation_history)
        
        print(f"LLM Output: {response}")  # Backend console logging
        
        # Parse the JSON response
        json_response = json.loads(response)
        
        # Check for NULL values
        null_message = check_null_values(json_response)
        audio_base64=None
        
        # Create audio response based on whether there are NULL values
        if null_message:
            audio_base64 = create_audio_response(null_message)
        # else:
        #     audio_base64 = create_audio_response("Your Event has been added at Google calendar")
        
        return {
            'json_response': response,
            'null_message': null_message,
            'audio': audio_base64
        }
    except Exception as e:
        logger.error(f"LLM Processing error: {e}")
        return str(e)

async def transcribe_audio_stream(websocket: WebSocket):    
    try:
        client = speech.SpeechAsyncClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            audio_channel_count=1,
            enable_separate_recognition_per_channel=False,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, 
            interim_results=True
        )

        async def request_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            
            buffer = bytearray()
            min_chunk_size = 1024
            connection_active = True

            while connection_active:
                try:
                    message = await websocket.receive()
                    
                    if 'bytes' in message:
                        chunk = message['bytes']
                        buffer.extend(chunk)
                        
                        if len(buffer) >= min_chunk_size:
                            yield speech.StreamingRecognizeRequest(audio_content=bytes(buffer))
                            buffer.clear()
                    
                    elif 'text' in message:
                        data = json.loads(message['text'])
                        if data.get('type') == 'COMMAND' and data.get('text') == 'STOP_RECORDING':
                            if buffer:
                                yield speech.StreamingRecognizeRequest(audio_content=bytes(buffer))
                            connection_active = False
                            break
                            
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                    connection_active = False
                    break
                except Exception as e:
                    logger.error(f"Error in request generator: {e}")
                    connection_active = False
                    break

        streaming_recognize = await client.streaming_recognize(requests=request_generator())
        
        try:
            async for response in streaming_recognize:
                if not response.results:
                    continue

                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        print(f"Transcription: {transcript}")
                        llm_response = await process_with_llm(transcript, websocket)
                        await websocket.send_json({
                            'type': 'FINAL_TRANSCRIPT',
                            'text': transcript,
                            'llm_response': llm_response['json_response'],
                            'null_message': llm_response['null_message'],
                            'audio': llm_response['audio']
                        })
                    else:
                        await websocket.send_json({
                            'type': 'INTERIM_TRANSCRIPT',
                            'text': transcript
                        })
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")

    except google_exceptions.InvalidArgument as e:
        logger.error(f"Invalid argument error: {e}")
        try:
            await websocket.send_json({
                'type': 'ERROR',
                'text': str(e)
            })
        except Exception:
            pass
    except Exception as e:
        logger.error(f"General error: {e}")
        try:
            await websocket.send_json({
                'type': 'ERROR',
                'text': str(e)
            })
        except Exception:
            pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        conversation_history = await manager.connect(websocket)
        while True:
            try:
                await transcribe_audio_stream(websocket)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in endpoint")
                manager.disconnect(websocket)
                break
            except Exception as e:
                logger.error(f"Error in websocket endpoint: {e}")
                # Don't continue on error, break the loop
                break
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)