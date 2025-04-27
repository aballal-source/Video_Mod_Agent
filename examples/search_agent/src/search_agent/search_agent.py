import logging
import os
from dotenv import load_dotenv
from src.search_agent.providers.model_provider import ModelProvider
from src.search_agent.providers.search_provider import SearchProvider
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler)
from typing import AsyncGenerator
from google.cloud import speech  # For transcription if you're using Google Cloud Speech-to-Text
import tempfile

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VideoProcessingAgent(AbstractAgent):
    def __init__(self, name: str):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        search_api_key = os.getenv("TAVILY_API_KEY")
        if not search_api_key:
            raise ValueError("TAVILY_API_KEY is not set") 
        self._search_provider = SearchProvider(api_key=search_api_key)

        google_cloud_speech_api_key = os.getenv("GOOGLE_CLOUD_SPEECH_API_KEY")
        if not google_cloud_speech_api_key:
            raise ValueError("GOOGLE_CLOUD_SPEECH_API_KEY is not set")
        self._speech_client = speech.SpeechClient()

    # Implement the assist method as required by the AbstractAgent class
    async def assist(self, session: Session, query: Query, response_handler: ResponseHandler):
        """Accepts MP4 files, transcribes them, summarizes."""
        mp4_file = query.attachments.get("mp4_file")
        if not mp4_file:
            await response_handler.emit_text_block("ERROR", "No MP4 file provided.")
            return

        await response_handler.emit_text_block("PROCESSING", "Processing MP4 file...")

        # 1. Transcribe the video
        transcription = await self.transcribe_video(mp4_file)

        # 2. Summarize the transcription
        summary = await self.summarize_text(transcription)

        # 3. Provide downloadable information
        await response_handler.emit_json("TRANSCRIPTION", {"text": transcription})
        await response_handler.emit_json("SUMMARY", {"summary": summary})

        await response_handler.complete()

    async def transcribe_video(self, mp4_file: str) -> str:
        """Transcribes the MP4 video file using Google Cloud Speech-to-Text."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            # Convert MP4 to audio (e.g., WAV or FLAC) using FFmpeg or another tool
            audio_path = temp_audio_file.name
            self.convert_mp4_to_audio(mp4_file, audio_path)
            
            # Use Google Cloud Speech API to transcribe the audio
            with open(audio_path, "rb") as audio_file:
                audio_content = audio_file.read()

            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )

            response = self._speech_client.recognize(config=config, audio=audio)

            # Combine transcriptions into a single string
            transcription = "\n".join([result.alternatives[0].transcript for result in response.results])
            return transcription

    def convert_mp4_to_audio(self, mp4_file: str, output_audio_path: str):
        """Converts MP4 video to audio (WAV or FLAC) using FFmpeg."""
        # Example: FFmpeg could be used for better audio extraction from MP4.
        # You can use the subprocess module to call FFmpeg from the command line:
        # subprocess.run(["ffmpeg", "-i", mp4_file, output_audio_path])
        pass

    async def summarize_text(self, text: str) -> AsyncGenerator[str, None]:
        """Summarizes the transcribed text using the model provider."""
        summary_query = f"Summarize the following text: {text}"
        async for chunk in self._model_provider.query_stream(summary_query):
            yield chunk

if __name__ == "__main__":
    # Create an instance of a VideoProcessingAgent
    agent = VideoProcessingAgent(name="Video Processing Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()
