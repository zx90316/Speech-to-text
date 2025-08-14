from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="vscc-faq",
      location="global",
  )

  audio1 = types.Part.from_uri(
      file_uri="gs://cloud-samples-data/generative-ai/audio/audio_transcription_data_commons.mp3",
      mime_type="audio/mpeg",
  )

  model = "gemini-2.5-flash-lite"
  contents = [
    types.Content(
      role="user",
      parts=[
        audio1,
        types.Part.from_text(text="""Generate a transcription of the audio, only extract speech and ignore background audio.""")
      ]
    )
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=0,
    ),
  )



  response = client.models.generate_content(
    model = model,
    contents = contents,
    config = generate_content_config,
    )

  print(response.text)

generate()