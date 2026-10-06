from dotenv import load_dotenv
import os, requests
load_dotenv()
res = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {os.getenv("GROQ_API_KEY")}'})
print(res.json())
