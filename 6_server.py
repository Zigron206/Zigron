import io
import json
import os
import time
import uvicorn
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, List
from dataclasses import dataclass

import struct
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Form
from google.cloud import speech_v1p1beta1 as speech
import asyncio

from gtts import gTTS
from openai import OpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from vector_database import DepressionKnowledgeBase
import json

# from vector_database import DocumentStore

# Set up Google Cloud credentials
load_dotenv()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Current/speech-to-text-denta.json"

app = FastAPI()

# Audio recording parameters
RATE = 16000

# Initialize thread pool for blocking operations
thread_pool = ThreadPoolExecutor(max_workers=10)  # One worker each for OpenAI, TTS, and other blocking ops

# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
def get_depression_info(query: str) -> dict:
    kb = DepressionKnowledgeBase()
    response = kb.query_knowledge_base(query)
    
    # Return only results without timestamp and other metadata
    simplified_response = {
        'results': response.get('results', {})
    }
    return simplified_response

# --------------------

# In-memory storage for the demographic data (global variable)
demographics_data = {}

class DemographicData(BaseModel):
    age: str
    gender: str
    maritalStatus: str
    ethnicity: str
    education: str
    occupation: str
    medicalHistory: str

@app.post("/demographics")
async def save_demographics(data: DemographicData):  # Change to use Pydantic model directly
    global demographics_data
    demographics_data = data.dict()
    return {
        "status": "success",
        "message": "Demographics saved successfully",
        "data": demographics_data
    }

# Create a single conversation history list that persists across all messages
conversation_history = []

async def generate_ai_response(transcript: str, openai_client: OpenAI) -> AsyncGenerator[str, None]:
    """Stream OpenAI API responses chunk by chunk with persistent conversation history"""
    start_time = time.perf_counter()
    try:
        # Add user message to the global conversation history
        conversation_history.append({'role': 'user', 'content': transcript})
        # result = get_depression_info(transcript)
        case_study="Case Study: Sarah, a 32-year-old accountant, presented with symptoms of Major Depressive Disorder (MDD), Moderate Severity, including persistent low mood, anhedonia, chronic fatigue, social withdrawal, and passive suicidal ideation lasting for over six months. Her symptoms began following the breakup of a five-year relationship, which triggered feelings of worthlessness, guilt, and hopelessness, compounded by work-related stress and social isolation. Sarah reported a history of perfectionism, self-criticism, and a traumatic childhood, including sexual abuse and growing up with a single, emotionally unavailable mother, which likely contributed to her vulnerability to depression. She also described insomnia alternating with hypersomnia, weight gain from emotional eating, and difficulty concentrating, resulting in declining work performance. Despite these challenges, Sarah displayed insight into her struggles, expressed interest in re-engaging with hobbies like hiking and reading, and was willing to pursue therapy, indicating a strong foundation for recovery. Her treatment plan includes Cognitive Behavioral Therapy (CBT) for cognitive restructuring and behavioral activation, trauma-focused therapy, and potential pharmacological support, alongside efforts to rebuild her social connections and address work stress. Regular monitoring and a biopsychosocial approach will guide her journey toward improved mental health and functionality."
        prompt = f"""
        A case study which you can use to cure: {case_study}
        --------------------------------------------------------
        Instructions:
        You are a mental health professional who is designed to conduct psychological analysis, 
        mental health examination and diagnose and give treatments as a therapist does while 
        maintaining empathy. You are a friendly voice bot that user would feel comfortable talking to.
        Follow the following guidelines:
            Your Core Behaviours:
            -Explore main reasons for seeking help.
            -Inquire about impact on daily life.
            -Note any specific triggers or patterns.
            -Ask one question at a time.
            -Wait for the client's response before proceeding.
            -Show empathy and active listening through appropriate acknowledgments.
            -Be open and understanding.
            -Flag any mentions of self-harm or suicide.
            -Maintain a professional, warm, and non-judgmental and humanoid tone.
            -Provide specific advice or diagnoses.
            Response Behaviour:
            -Validate feelings, Example: "I hear how difficult this has been for you..."
            -Show understanding, Example: "It makes sense that you would feel that way..."
            -Encourage elaboration, Example: "Could you tell me more about..."
            -Express empathy, Example: "That sounds really challenging..."
            -Reflect back what the user has said: “So what I am understanding is...”
            Critical situations:
            If client mentions active suicidal thoughts or current self-harm, respond with 
            "I am not equipped to help with that please call suicide prevention hotline immediately"
            End of Session:
            -Summarize key points discussed.
            -Provide appropriate resources.
            -Clear next steps.
            """
            
        # Initialize system prompt if needed (only once)
        if not any('system' in msg for msg in conversation_history):
            conversation_history.insert(0, {'role': 'system', 'content': prompt})
        
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(
            thread_pool,
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=conversation_history,
                stream=True
            )
        )
        
        full_response = []
        
        async def async_iteration():
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response.append(content)
                        yield content
                await asyncio.sleep(0)
        
        async for content in async_iteration():
            yield content
        
        # Add assistant's response to conversation history
        conversation_history.append({
            'role': 'assistant',
            'content': ''.join(full_response)
        })
        
        # Optional: Limit conversation history size to prevent memory issues
        if len(conversation_history) > 20:  # Keep last 20 messages
            # Keep system prompt and last 19 messages
            system_prompt = conversation_history[0]
            conversation_history.clear()
            conversation_history.append(system_prompt)
            conversation_history.extend(conversation_history[-19:])
        
    except Exception as e:
        print(f"Error in streaming response: {e}")
        raise
    finally:
        end_time = time.perf_counter()
        print('--------------------------------------------------------------')
        print(f"generate_ai_response took {end_time - start_time:.4f} seconds")
        print('--------------------------------------------------------------')

async def generate_audio_response(text: str):
    """Run text-to-speech conversion in thread pool to avoid blocking"""
    loop = asyncio.get_running_loop()

    def _generate_audio():
        # Use OpenAI's text-to-speech API
        response = openai_client.audio.speech.create(
            model="tts-1",  # Use the OpenAI tts model
            voice="alloy",  # You can customize the voice here
            input=text       # Pass the text input
        )
        
        # Get the raw audio content as bytes
        audio_content = response.content
        
        # Create BytesIO object to store the audio data
        audio_fp = io.BytesIO()
        audio_fp.write(audio_content)
        audio_fp.seek(0)
        return audio_fp.getvalue()

    return await loop.run_in_executor(thread_pool, _generate_audio)

# async def generate_audio_response(text: str):
#     """Run text-to-speech conversion in thread pool to avoid blocking"""
#     loop = asyncio.get_running_loop()

#     def _generate_audio():
#         tts = gTTS(text, lang='en')
#         audio_fp = io.BytesIO()
#         tts.write_to_fp(audio_fp)
#         audio_fp.seek(0)
#         return audio_fp.getvalue()

#     return await loop.run_in_executor(thread_pool, _generate_audio)


# Suppress specific spaCy UserWarnings related to lemmatizer and POS tagging
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)

#------------------------------------------------------

# class TextBuffer:
#     def __init__(self):
#         self.current_buffer = []
#         self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser'])  # Minimal processing
#         self.nlp.add_pipe('sentencizer')  # Add lightweight sentencizer for sentence boundary detection

#     def is_potential_sentence_end(self, chunk: str) -> bool:
#         """Check if the chunk is a sentence-ending punctuation."""
#         return chunk in {'.', '!', '?'}

#     def add_chunk_and_get_sentences(self, new_chunk: str) -> List[str]:
#         """Process token and return complete sentences when punctuation is detected."""

#         # Skip processing if the new_chunk is empty
#         if not new_chunk.strip():
#             return []

#         # If new_chunk is a sentence-ending punctuation, append it directly to the last word
#         if self.is_potential_sentence_end(new_chunk) and self.current_buffer:
#             self.current_buffer[-1] += new_chunk
#         else:
#             self.current_buffer.append(new_chunk)

#         # Check if we have potential sentence-ending punctuation
#         if not any(self.is_potential_sentence_end(token[-1]) for token in self.current_buffer):
#             return []

#         # Join buffer and check for complete sentences
#         text = " ".join(self.current_buffer).strip()
#         doc = self.nlp(text)
#         sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

#         # Clear buffer after sentences are processed
#         if sentences:
#             self.current_buffer.clear()

#         return sentences

#     def get_remaining_text(self) -> str:
#         """Get any remaining text when stream ends."""
#         return " ".join(self.current_buffer).strip()

#---------------------------------------------------------

class MessageSequencer:
    def __init__(self, message_seq):
        self.message_seq = message_seq
        self.chunk_seq = 0
        self.send_response_audio = True  # Control for sending audio back

    def next_chunk(self):
        """Increment and return the chunk sequence number."""
        self.chunk_seq += 1
        return self.chunk_seq


async def transcribe_audio_stream(websocket: WebSocket):
    client = speech.SpeechAsyncClient()
    active_sequences = {}  # Dictionary to keep track of active sequences
    active_sequences_number = 1

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    async def request_generator():
        print("Yielding StreamingRecognizeRequest with 'streaming_config'.")
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)

        while True:
            try:
                # Receive raw message first
                message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
                # print("Stream data received.")

                # Check message type
                if 'bytes' in message:
                    # Handle audio data
                    yield speech.StreamingRecognizeRequest(audio_content=message['bytes'])

                elif 'text' in message:  # Handle text message
                    try:
                        data = json.loads(message['text'])
                        print(f"Received text message: {data}")

                        # Check for stop command
                        if data.get('type') == 'COMMAND' and data.get('text') == 'CLOSE_WEBSOCKET':
                            print("WebSocket disconnection command received.")
                            break

                        # Process other text messages as needed
                        if data.get('type') == 'message':
                            print(f"Message content: {data.get('text')}")

                    except json.JSONDecodeError:
                        print("Error decoding JSON message")
                        continue

            except asyncio.TimeoutError:
                print("Timeout waiting for data")
                continue

            except Exception as e:
                print(f"Error in request generator: {e}")
                break

    async def process_final_transcript(transcript: str, openai_client: OpenAI, sequencer: MessageSequencer):
        try:
            transcript = transcript.strip()

            await websocket.send_json({
                'messageSeq': sequencer.message_seq,
                'chunkSeq': sequencer.chunk_seq,
                'type': 'MESSAGE',
                'text': f"MESSAGE, messageSeq: {sequencer.message_seq}, chunkSeq: {sequencer.chunk_seq}, transcript: {transcript}",
            })

            # if "exit" in transcript.lower() or "quit" in transcript.lower():
            #     await websocket.send_json({
            #         'type': 'COMMAND',
            #         'text': "DISCONNECT_STREAM"
            #     })
            #     return

            complete_response = ""
            buffer_text = ""  # Buffer to accumulate text

            async for response_chunk in generate_ai_response(transcript, openai_client):
                complete_response = complete_response + " " + response_chunk
                buffer_text += response_chunk

                # Send text immediately for responsive UI
                await websocket.send_json({
                        'messageSeq': sequencer.message_seq,
                    'chunkSeq': sequencer.next_chunk(),
                    'type': 'MESSAGE',
                    'text': f"MESSAGE, messageSeq: {sequencer.message_seq}, chunkSeq: {sequencer.chunk_seq}, CHUNK: {response_chunk}",
                })

                # Generate audio only when buffer has accumulated enough text (e.g., 10 words)
                if len(buffer_text.split()) >= 7 and sequencer.send_response_audio:
                    audio_data = await generate_audio_response(buffer_text.strip())
                    metadata = struct.pack('!II', sequencer.message_seq, sequencer.chunk_seq)
                    combined_data = metadata + audio_data
                    await websocket.send_bytes(combined_data)
                    buffer_text = ""  # Clear buffer after sending

            # Send any remaining buffered text as audio
            if buffer_text.strip() and sequencer.send_response_audio:
                audio_data = await generate_audio_response(buffer_text.strip())
                metadata = struct.pack('!II', sequencer.message_seq, sequencer.chunk_seq)
                combined_data = metadata + audio_data
                await websocket.send_bytes(combined_data)

        except Exception as e:
            print(f"Error in response generation: {e}")
            await websocket.send_json({
                'type': 'ERROR',
                'text': str(e)
            })
    try:
        streaming_recognize = await client.streaming_recognize(requests=request_generator())
        sentence_completed = False

        async for response in streaming_recognize:
            for result in response.results:
                # print(f"Result: {result}")
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                if result.is_final:
                    if transcript:
                        # print(transcript)
                        sentence_completed = True
                        # New sequencer for each final transcript
                        sequencer = MessageSequencer(message_seq=active_sequences_number)
                        # Add new sequence to active_sequences
                        active_sequences[active_sequences_number] = sequencer
                        active_sequences_number += 1
                        # await websocket.send_text(f"Final: {transcript}")
                        # Process final transcript in parallel with continued listening
                        asyncio.create_task(process_final_transcript(transcript, openai_client, sequencer))

                else:
                    if transcript:
                        if sentence_completed:
                            # Send signal new sentence is started so stop if client is playing anything
                            sentence_completed = False
                            # Disable send_response_audio for all previous sequences
                            for seq in active_sequences.values():
                                seq.send_response_audio = False
                            await websocket.send_json({
                                'type': 'COMMAND',
                                'text': "STOP_SPEAKING"
                            })
                            print(f"COMMAND, STOP_SPEAKING")
                            print(f"Stopping further audio response for old messages.")

                        # await websocket.send_text(f"Interim: {transcript}")
                        # await websocket.send_json({
                        #     'type': 'MESSAGE',
                        #     'text': f"Interim: {transcript}"
                        # })

    except Exception as e:
        print(f"Error in transcription: {e}")
        await websocket.send_text(f"Error: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        await transcribe_audio_stream(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5)