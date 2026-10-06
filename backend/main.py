import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from fastapi.responses import FileResponse

# Load environment variables (like GROQ_API_KEY) from .env file
load_dotenv()

# Initialize Groq API client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the LLM model to use for completion tasks
model = "openai/gpt-oss-120b"

# Initialize FastAPI application instance
app = FastAPI()

# Add CORS Middleware immediately after initializing app so browser fetch calls don't get blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route serving your frontend index.html from the frontend folder
@app.get("/")
def home():
    # Navigates up from backend folder to hiremeai, then into frontend/index.html
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(html_path)

# Root health check endpoint returning status
@app.get("/health")
def health():
    return {
        "status": "online",
        "message": "FastAPI backend is running successfully!"
    }

# Pydantic model for single job or internship experience entry
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

# Pydantic model representing structured candidate resume data
class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

# Generate JSON schema from the Resume model to guide the LLM parser
resume_schema = Resume.model_json_schema()

# Pydantic model for incoming chat request payload
class ChatRequest(BaseModel):
    question: str

# Pydantic model for job matching request
class JobMatchRequest(BaseModel):
    job_description: str

# Sends candidate questions to Groq LLM to act as the candidate during HR interviews
def ask_candidate(question: str, resume: Resume):
    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say "I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing this candidate.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content

# Parses raw resume text into a structured Resume Pydantic model using Groq LLM
def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system = {
        "role": "system",
        "content": system_prompt
    }
    message_user = {
        "role": "user",
        "content": user_prompt
    }
    messages = [message_system, message_user]
    response_format = {
        "type": "json_object"
    }
    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume

# Extracts plain text content page by page from a local PDF file path
def read_pdf(file_path: Path):
    if not file_path.exists():
        return "No resume available."
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Chat endpoint that reads resume PDF, parses it, answers via LLM, and includes screening=True
@app.post("/chat")
def chat(request: ChatRequest):
    resume_text = read_pdf(Path("my_resume.pdf"))
    if resume_text == "No resume available.":
        return {"screening": False, "answer": "Resume PDF not found. Please ensure 'my_resume.pdf' exists in the backend directory."}
    
    resume = parse_resume(resume_text)
    answer = ask_candidate(request.question, resume)
    return {
        "screening": True,
        "answer": answer
    }

# Job Matching Endpoint
@app.post("/match_job")
def match_job(request: JobMatchRequest):
    resume_text = read_pdf(Path("my_resume.pdf"))
    if resume_text == "No resume available.":
        return {"analysis": "Resume PDF not found. Please ensure 'my_resume.pdf' exists in the backend directory."}
        
    resume = parse_resume(resume_text)
    
    system_prompt = f"""
    You are an expert technical recruiter and career coach.
    
    Below is the candidate's parsed resume:
    {resume.model_dump_json(indent=2)}
    
    The candidate wants to know how well they fit the following Job Description:
    {request.job_description}
    
    Analyze the match between the resume and the job description.
    Provide your analysis in Markdown format using the exact sections below.
    CRITICAL: You MUST use proper newlines for each bullet point. Do not put everything in one paragraph.
    
    ### Match Score
    Provide a percentage match score based on skills and experience fit.
    
    ### Strengths & Matching Skills
    * **Languages:** (List languages here)
    * **Frameworks:** (List frameworks here)
    * **Tools/Databases:** (List tools here)
    * **Soft Skills:** (List soft skills here)
    
    ### Gaps & Missing Skills
    * **Missing Tech Skills:** (List here)
    * **Missing Experience:** (List here)
    
    ### Recommendation
    Give a short, 2-sentence recommendation on whether they should apply or what they should focus on learning next.
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please analyze my fit for this job."}
        ]
    )
    
    return {"analysis": response.choices[0].message.content}