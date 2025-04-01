from google import genai  # Google Generative AI SDK (google-genai)

# Initialize Gemini API client with your API key
client = genai.Client(api_key="AIzaSyBXQFQ4PbB29BteSFs1zDq5dD8o8YkbKxg")

# Send a single text prompt to Gemini 2.0 Flash
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)
