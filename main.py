# main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from bson.errors import InvalidId
import jwt
import bcrypt
import os
import uuid
import time
import motor.motor_asyncio
from email_validator import validate_email, EmailNotValidError
import aiohttp
import asyncio
import threading
import logging
import uvicorn
from dotenv import load_dotenv
import re
import mimetypes
import math
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Student Dashboard API",
    description="Backend API for Student Dashboard Application",
    version="1.0.0"
)

# Add rate limiting handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # In production, replace with specific hosts
)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:password@cluster.mongodb.net/studentdashboard?retryWrites=true&w=majority")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.studentdashboard

# Create indexes for better performance
async def create_indexes():
    await db.users.create_index([("email", ASCENDING)], unique=True)
    await db.assignments.create_index([("user_id", ASCENDING)])
    await db.events.create_index([("user_id", ASCENDING)])
    await db.study_sessions.create_index([("user_id", ASCENDING)])
    await db.materials.create_index([("user_id", ASCENDING)])
    await db.subjects.create_index([("user_id", ASCENDING)])
    await db.goals.create_index([("user_id", ASCENDING)])
    await db.notes.create_index([("user_id", ASCENDING)])

# JWT Settings
JWT_SECRET = os.getenv("JWT_SECRET", "your_jwt_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60 * 24  # 24 hours

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# File upload settings
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".xlsx", ".xls", ".doc", ".docx", ".txt"
}
MAX_FILES_PER_USER = 10

# Self-ping mechanism to prevent Render from shutting down
async def ping_server():
    app_url = os.getenv("APP_URL", "https://your-app-url.onrender.com")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{app_url}/health") as response:
                logger.info(f"Self-ping result: {response.status}")
        except Exception as e:
            logger.error(f"Self-ping failed: {str(e)}")

async def start_ping_task():
    while True:
        await ping_server()
        await asyncio.sleep(60 * 14)  # Ping every 14 minutes (Render free tier sleeps after 15 min)

# Start ping task in background
@app.on_event("startup")
async def startup_event():
    await create_indexes()
    asyncio.create_task(start_ping_task())

# Pydantic Models
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            if not isinstance(v, str) and not isinstance(v, ObjectId):
                raise ValueError("Not a valid ObjectId")
            return str(v)
        except InvalidId:
            raise ValueError("Not a valid ObjectId")

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr
    name: str
    created_at: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None

class SubjectCreate(BaseModel):
    name: str
    color: str = "#4287f5"  # Default blue color
    description: Optional[str] = None

class SubjectResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str
    color: str
    description: Optional[str] = None
    user_id: PyObjectId

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

class AssignmentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: datetime
    subject_id: PyObjectId
    priority: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ["low", "medium", "high"]:
            raise ValueError('Priority must be low, medium, or high')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ["pending", "in_progress", "completed"]:
            raise ValueError('Status must be pending, in_progress, or completed')
        return v

class AssignmentResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: Optional[str] = None
    due_date: datetime
    subject_id: PyObjectId
    priority: str
    status: str
    user_id: PyObjectId
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class AssignmentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    subject_id: Optional[PyObjectId] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    
    @validator('priority')
    def validate_priority(cls, v):
        if v is not None and v not in ["low", "medium", "high"]:
            raise ValueError('Priority must be low, medium, or high')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None and v not in ["pending", "in_progress", "completed"]:
            raise ValueError('Status must be pending, in_progress, or completed')
        return v

class EventCreate(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    type: str = "personal"  # exam, holiday, personal
    subject_id: Optional[PyObjectId] = None
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ["exam", "holiday", "personal"]:
            raise ValueError('Type must be exam, holiday, or personal')
        return v
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v < values['start_time']:
            raise ValueError('End time must be after start time')
        return v

class EventResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    type: str
    subject_id: Optional[PyObjectId] = None
    user_id: PyObjectId
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class EventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    type: Optional[str] = None
    subject_id: Optional[PyObjectId] = None
    
    @validator('type')
    def validate_type(cls, v):
        if v is not None and v not in ["exam", "holiday", "personal"]:
            raise ValueError('Type must be exam, holiday, or personal')
        return v

class StudySessionCreate(BaseModel):
    subject_id: PyObjectId
    planned_duration: int  # in minutes
    description: Optional[str] = None
    scheduled_date: datetime
    use_pomodoro: bool = False
    pomodoro_work: int = 25  # Default 25 minutes
    pomodoro_break: int = 5  # Default 5 minutes
    
    @validator('planned_duration')
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v
    
    @validator('pomodoro_work')
    def validate_pomodoro_work(cls, v, values):
        if values.get('use_pomodoro', False) and (v < 5 or v > 60):
            raise ValueError('Pomodoro work time must be between 5 and 60 minutes')
        return v
    
    @validator('pomodoro_break')
    def validate_pomodoro_break(cls, v, values):
        if values.get('use_pomodoro', False) and (v < 1 or v > 30):
            raise ValueError('Pomodoro break time must be between 1 and 30 minutes')
        return v

class StudySessionResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    subject_id: PyObjectId
    planned_duration: int
    actual_duration: Optional[int] = None
    description: Optional[str] = None
    scheduled_date: datetime
    completed: bool = False
    completed_at: Optional[datetime] = None
    use_pomodoro: bool
    pomodoro_work: int
    pomodoro_break: int
    user_id: PyObjectId
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class StudySessionUpdate(BaseModel):
    subject_id: Optional[PyObjectId] = None
    planned_duration: Optional[int] = None
    actual_duration: Optional[int] = None
    description: Optional[str] = None
    scheduled_date: Optional[datetime] = None
    completed: Optional[bool] = None
    completed_at: Optional[datetime] = None
    use_pomodoro: Optional[bool] = None
    pomodoro_work: Optional[int] = None
    pomodoro_break: Optional[int] = None

class MaterialResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str
    file_type: str
    file_size: int
    subject_id: PyObjectId
    description: Optional[str] = None
    file_path: str
    uploaded_at: datetime
    user_id: PyObjectId

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MaterialUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    subject_id: Optional[PyObjectId] = None

class GoalCreate(BaseModel):
    title: str
    description: Optional[str] = None
    target_date: datetime
    subject_id: Optional[PyObjectId] = None
    milestones: List[Dict[str, Any]] = []
    
    @validator('target_date')
    def validate_target_date(cls, v):
        if v < datetime.now():
            raise ValueError('Target date must be in the future')
        return v

class GoalResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: Optional[str] = None
    target_date: datetime
    subject_id: Optional[PyObjectId] = None
    milestones: List[Dict[str, Any]]
    progress: int = 0
    completed: bool = False
    completed_at: Optional[datetime] = None
    user_id: PyObjectId
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class GoalUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_date: Optional[datetime] = None
    subject_id: Optional[PyObjectId] = None
    milestones: Optional[List[Dict[str, Any]]] = None
    progress: Optional[int] = None
    completed: Optional[bool] = None
    completed_at: Optional[datetime] = None

class NoteCreate(BaseModel):
    title: str
    content: str
    subject_id: Optional[PyObjectId] = None
    tags: List[str] = []

class NoteResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    content: str
    subject_id: Optional[PyObjectId] = None
    tags: List[str]
    user_id: PyObjectId
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    subject_id: Optional[PyObjectId] = None
    tags: Optional[List[str]] = None

class StatisticsResponse(BaseModel):
    total_assignments: int
    completed_assignments: int
    pending_assignments: int
    upcoming_events: int
    study_hours: float
    subject_stats: List[Dict[str, Any]]
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Authentication Functions
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        token_data = TokenData(email=email, user_id=user_id)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"_id": ObjectId(token_data.user_id)})
    if user is None:
        raise credentials_exception
    return user

# Authentication endpoints
@app.post("/register", response_model=UserResponse, status_code=201)
@limiter.limit("10/minute")
async def register_user(request: Request, user: UserCreate):
    try:
        # Validate email
        validate_email(user.email)
    except EmailNotValidError:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Create user document
    new_user = {
        "email": user.email,
        "password": hashed_password,
        "name": user.name,
        "created_at": datetime.utcnow()
    }
    
    # Insert into database
    result = await db.users.insert_one(new_user)
    
    # Create default subjects
    default_subjects = [
        {"name": "Mathematics", "color": "#ff6b6b", "description": "All math related courses", "user_id": result.inserted_id},
        {"name": "Science", "color": "#48dbfb", "description": "Physics, Chemistry, Biology", "user_id": result.inserted_id},
        {"name": "English", "color": "#1dd1a1", "description": "Literature and language studies", "user_id": result.inserted_id},
        {"name": "History", "color": "#feca57", "description": "Historical studies", "user_id": result.inserted_id},
    ]
    await db.subjects.insert_many(default_subjects)
    
    # Return created user without password
    created_user = await db.users.find_one({"_id": result.inserted_id})
    created_user.pop("password")
    return created_user

@app.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Find user by email
    user = await db.users.find_one({"email": form_data.username})
    
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    user_data = {**current_user}
    user_data.pop("password", None)
    return user_data

# Subject endpoints
@app.post("/subjects", response_model=SubjectResponse, status_code=201)
@limiter.limit("30/minute")
async def create_subject(request: Request, subject: SubjectCreate, current_user: dict = Depends(get_current_user)):
    new_subject = {
        **subject.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.subjects.insert_one(new_subject)
    created_subject = await db.subjects.find_one({"_id": result.inserted_id})
    
    return created_subject

@app.get("/subjects", response_model=List[SubjectResponse])
@limiter.limit("60/minute")
async def get_subjects(
    request: Request,
    skip: int = 0, 
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    subjects = await db.subjects.find(
        {"user_id": ObjectId(current_user["_id"])}
    ).skip(skip).limit(limit).to_list(limit)
    
    return subjects

@app.get("/subjects/{subject_id}", response_model=SubjectResponse)
@limiter.limit("60/minute")
async def get_subject(
    request: Request,
    subject_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        subject = await db.subjects.find_one({
            "_id": ObjectId(subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    return subject

@app.put("/subjects/{subject_id}", response_model=SubjectResponse)
@limiter.limit("30/minute")
async def update_subject(
    request: Request,
    subject_id: str,
    subject_update: SubjectUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if subject exists and belongs to user
        subject = await db.subjects.find_one({
            "_id": ObjectId(subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Update only provided fields
    update_data = {k: v for k, v in subject_update.dict().items() if v is not None}
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.subjects.update_one(
            {"_id": ObjectId(subject_id)},
            {"$set": update_data}
        )
    
    # Return updated subject
    updated_subject = await db.subjects.find_one({"_id": ObjectId(subject_id)})
    return updated_subject

@app.delete("/subjects/{subject_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_subject(
    request: Request,
    subject_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if subject exists and belongs to user
        subject = await db.subjects.find_one({
            "_id": ObjectId(subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Delete subject
    await db.subjects.delete_one({"_id": ObjectId(subject_id)})
    
    # Update related records to remove this subject
    await db.assignments.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    await db.events.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    await db.study_sessions.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    await db.materials.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    await db.goals.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    await db.notes.update_many(
        {"subject_id": ObjectId(subject_id)},
        {"$set": {"subject_id": None}}
    )
    
    return None

# Assignment endpoints
@app.post("/assignments", response_model=AssignmentResponse, status_code=201)
@limiter.limit("30/minute")
async def create_assignment(
    request: Request,
    assignment: AssignmentCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id exists and belongs to user
    try:
        subject = await db.subjects.find_one({
            "_id": ObjectId(assignment.subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    new_assignment = {
        **assignment.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.assignments.insert_one(new_assignment)
    created_assignment = await db.assignments.find_one({"_id": result.inserted_id})
    
    return created_assignment

@app.get("/assignments", response_model=List[AssignmentResponse])
@limiter.limit("60/minute")
async def get_assignments(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    subject_id: Optional[str] = None,
    priority: Optional[str] = None,
    due_before: Optional[datetime] = None,
    due_after: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if status:
        if status not in ["pending", "in_progress", "completed"]:
            raise HTTPException(status_code=400, detail="Invalid status value")
        query["status"] = status
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if priority:
        if priority not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Invalid priority value")
        query["priority"] = priority
    
    date_query = {}
    if due_before:
        date_query["$lte"] = due_before
    if due_after:
        date_query["$gte"] = due_after
    
    if date_query:
        query["due_date"] = date_query
    
    # Get assignments with filters
    assignments = await db.assignments.find(query).sort("due_date", 1).skip(skip).limit(limit).to_list(limit)
    
    return assignments

@app.get("/assignments/{assignment_id}", response_model=AssignmentResponse)
@limiter.limit("60/minute")
async def get_assignment(
    request: Request,
    assignment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        assignment = await db.assignments.find_one({
            "_id": ObjectId(assignment_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid assignment ID format")
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    return assignment

@app.put("/assignments/{assignment_id}", response_model=AssignmentResponse)
@limiter.limit("30/minute")
async def update_assignment(
    request: Request,
    assignment_id: str,
    assignment_update: AssignmentUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if assignment exists and belongs to user
        assignment = await db.assignments.find_one({
            "_id": ObjectId(assignment_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid assignment ID format")
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in assignment_update.dict().items() if v is not None}
    if "subject_id" in update_data:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Update assignment
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.assignments.update_one(
            {"_id": ObjectId(assignment_id)},
            {"$set": update_data}
        )
    
    # Return updated assignment
    updated_assignment = await db.assignments.find_one({"_id": ObjectId(assignment_id)})
    return updated_assignment

@app.delete("/assignments/{assignment_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_assignment(
    request: Request,
    assignment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if assignment exists and belongs to user
        assignment = await db.assignments.find_one({
            "_id": ObjectId(assignment_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid assignment ID format")
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    # Delete assignment
    await db.assignments.delete_one({"_id": ObjectId(assignment_id)})
    return None

# Event endpoints
@app.post("/events", response_model=EventResponse, status_code=201)
@limiter.limit("30/minute")
async def create_event(
    request: Request,
    event: EventCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id if provided
    if event.subject_id:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(event.subject_id),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    new_event = {
        **event.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.events.insert_one(new_event)
    created_event = await db.events.find_one({"_id": result.inserted_id})
    
    return created_event

@app.get("/events", response_model=List[EventResponse])
@limiter.limit("60/minute")
async def get_events(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    type: Optional[str] = None,
    subject_id: Optional[str] = None,
    start_after: Optional[datetime] = None,
    start_before: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if type:
        if type not in ["exam", "holiday", "personal"]:
            raise HTTPException(status_code=400, detail="Invalid event type")
        query["type"] = type
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    date_query = {}
    if start_after:
        date_query["$gte"] = start_after
    if start_before:
        date_query["$lte"] = start_before
    
    if date_query:
        query["start_time"] = date_query
    
    # Get events with filters
    events = await db.events.find(query).sort("start_time", 1).skip(skip).limit(limit).to_list(limit)
    
    return events

@app.get("/events/{event_id}", response_model=EventResponse)
@limiter.limit("60/minute")
async def get_event(
    request: Request,
    event_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        event = await db.events.find_one({
            "_id": ObjectId(event_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid event ID format")
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return event

@app.put("/events/{event_id}", response_model=EventResponse)
@limiter.limit("30/minute")
async def update_event(
    request: Request,
    event_id: str,
    event_update: EventUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if event exists and belongs to user
        event = await db.events.find_one({
            "_id": ObjectId(event_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid event ID format")
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in event_update.dict().items() if v is not None}
    if "subject_id" in update_data and update_data["subject_id"]:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Validate start/end time logic if both are provided
    if "start_time" in update_data and "end_time" in update_data:
        if update_data["end_time"] < update_data["start_time"]:
            raise HTTPException(status_code=400, detail="End time must be after start time")
    elif "start_time" in update_data:
        # Only start_time is being updated, check against existing end_time
        if update_data["start_time"] > event["end_time"]:
            raise HTTPException(status_code=400, detail="Start time must be before end time")
    elif "end_time" in update_data:
        # Only end_time is being updated, check against existing start_time
        if update_data["end_time"] < event["start_time"]:
            raise HTTPException(status_code=400, detail="End time must be after start time")
    
    # Update event
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.events.update_one(
            {"_id": ObjectId(event_id)},
            {"$set": update_data}
        )
    
    # Return updated event
    updated_event = await db.events.find_one({"_id": ObjectId(event_id)})
    return updated_event

@app.delete("/events/{event_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_event(
    request: Request,
    event_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if event exists and belongs to user
        event = await db.events.find_one({
            "_id": ObjectId(event_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid event ID format")
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Delete event
    await db.events.delete_one({"_id": ObjectId(event_id)})
    return None

# Study Session endpoints
@app.post("/study-sessions", response_model=StudySessionResponse, status_code=201)
@limiter.limit("30/minute")
async def create_study_session(
    request: Request,
    study_session: StudySessionCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id
    try:
        subject = await db.subjects.find_one({
            "_id": ObjectId(study_session.subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    new_study_session = {
        **study_session.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "completed": False,
        "actual_duration": None,
        "completed_at": None,
        "created_at": datetime.utcnow()
    }
    
    result = await db.study_sessions.insert_one(new_study_session)
    created_study_session = await db.study_sessions.find_one({"_id": result.inserted_id})
    
    return created_study_session

@app.get("/study-sessions", response_model=List[StudySessionResponse])
@limiter.limit("60/minute")
async def get_study_sessions(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[str] = None,
    completed: Optional[bool] = None,
    scheduled_after: Optional[datetime] = None,
    scheduled_before: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if completed is not None:
        query["completed"] = completed
    
    date_query = {}
    if scheduled_after:
        date_query["$gte"] = scheduled_after
    if scheduled_before:
        date_query["$lte"] = scheduled_before
    
    if date_query:
        query["scheduled_date"] = date_query
    
    # Get study sessions with filters
    study_sessions = await db.study_sessions.find(query).sort("scheduled_date", 1).skip(skip).limit(limit).to_list(limit)
    
    return study_sessions

@app.get("/study-sessions/{session_id}", response_model=StudySessionResponse)
@limiter.limit("60/minute")
async def get_study_session(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        study_session = await db.study_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid study session ID format")
    
    if not study_session:
        raise HTTPException(status_code=404, detail="Study session not found")
    
    return study_session

@app.put("/study-sessions/{session_id}", response_model=StudySessionResponse)
@limiter.limit("30/minute")
async def update_study_session(
    request: Request,
    session_id: str,
    session_update: StudySessionUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if study session exists and belongs to user
        study_session = await db.study_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid study session ID format")
    
    if not study_session:
        raise HTTPException(status_code=404, detail="Study session not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in session_update.dict().items() if v is not None}
    if "subject_id" in update_data:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Handle completed logic
    if "completed" in update_data and update_data["completed"] and not study_session["completed"]:
        # Mark as completed now if not already completed
        if "completed_at" not in update_data or update_data["completed_at"] is None:
            update_data["completed_at"] = datetime.utcnow()
    
    # Update study session
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.study_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": update_data}
        )
    
    # Return updated study session
    updated_session = await db.study_sessions.find_one({"_id": ObjectId(session_id)})
    return updated_session

@app.delete("/study-sessions/{session_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_study_session(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if study session exists and belongs to user
        study_session = await db.study_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid study session ID format")
    
    if not study_session:
        raise HTTPException(status_code=404, detail="Study session not found")
    
    # Delete study session
    await db.study_sessions.delete_one({"_id": ObjectId(session_id)})
    return None

# Materials (File Upload) endpoints
@app.post("/materials", response_model=MaterialResponse, status_code=201)
@limiter.limit("10/minute")
async def upload_material(
    request: Request,
    subject_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id
    try:
        subject = await db.subjects.find_one({
            "_id": ObjectId(subject_id),
            "user_id": ObjectId(current_user["_id"])
        })
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Check user's file count limit
    material_count = await db.materials.count_documents({"user_id": ObjectId(current_user["_id"])})
    if material_count >= MAX_FILES_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"You've reached the maximum limit of {MAX_FILES_PER_USER} files. Please delete some files before uploading more."
        )
    
    # Validate file size
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the maximum limit of 5MB"
        )
    
    # Get file extension and validate
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # In a real production environment, you'd store this in a cloud storage
    # For this example, we'll simulate file storage by recording metadata
    file_path = f"uploads/{unique_filename}"
    
    # Determine file type category
    file_type = "document"
    if file_ext in [".jpg", ".jpeg", ".png", ".gif"]:
        file_type = "image"
    elif file_ext in [".xlsx", ".xls"]:
        file_type = "spreadsheet"
    elif file_ext in [".pdf"]:
        file_type = "pdf"
    
    # Create material record
    new_material = {
        "name": name,
        "description": description,
        "file_type": file_type,
        "file_size": file_size,
        "file_path": file_path,
        "subject_id": ObjectId(subject_id),
        "user_id": ObjectId(current_user["_id"]),
        "uploaded_at": datetime.utcnow()
    }
    
    result = await db.materials.insert_one(new_material)
    created_material = await db.materials.find_one({"_id": result.inserted_id})
    
    return created_material

@app.get("/materials", response_model=List[MaterialResponse])
@limiter.limit("60/minute")
async def get_materials(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[str] = None,
    file_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if file_type:
        if file_type not in ["document", "image", "spreadsheet", "pdf"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
        query["file_type"] = file_type
    
    # Get materials with filters
    materials = await db.materials.find(query).sort("uploaded_at", -1).skip(skip).limit(limit).to_list(limit)
    
    return materials

@app.get("/materials/{material_id}", response_model=MaterialResponse)
@limiter.limit("60/minute")
async def get_material(
    request: Request,
    material_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        material = await db.materials.find_one({
            "_id": ObjectId(material_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid material ID format")
    
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    return material

@app.put("/materials/{material_id}", response_model=MaterialResponse)
@limiter.limit("30/minute")
async def update_material(
    request: Request,
    material_id: str,
    material_update: MaterialUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if material exists and belongs to user
        material = await db.materials.find_one({
            "_id": ObjectId(material_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid material ID format")
    
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in material_update.dict().items() if v is not None}
    if "subject_id" in update_data:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Update material
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.materials.update_one(
            {"_id": ObjectId(material_id)},
            {"$set": update_data}
        )
    
    # Return updated material
    updated_material = await db.materials.find_one({"_id": ObjectId(material_id)})
    return updated_material

@app.delete("/materials/{material_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_material(
    request: Request,
    material_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if material exists and belongs to user
        material = await db.materials.find_one({
            "_id": ObjectId(material_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid material ID format")
    
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    # Delete material
    await db.materials.delete_one({"_id": ObjectId(material_id)})
    
    # In a real production environment, you'd also delete the file from storage
    
    return None

# Goal endpoints
@app.post("/goals", response_model=GoalResponse, status_code=201)
@limiter.limit("30/minute")
async def create_goal(
    request: Request,
    goal: GoalCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id if provided
    if goal.subject_id:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(goal.subject_id),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Process milestones
    for milestone in goal.milestones:
        if "completed" not in milestone:
            milestone["completed"] = False
    
    new_goal = {
        **goal.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "progress": 0,
        "completed": False,
        "completed_at": None,
        "created_at": datetime.utcnow()
    }
    
    result = await db.goals.insert_one(new_goal)
    created_goal = await db.goals.find_one({"_id": result.inserted_id})
    
    return created_goal

@app.get("/goals", response_model=List[GoalResponse])
@limiter.limit("60/minute")
async def get_goals(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[str] = None,
    completed: Optional[bool] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if completed is not None:
        query["completed"] = completed
    
    # Get goals with filters
    goals = await db.goals.find(query).sort("target_date", 1).skip(skip).limit(limit).to_list(limit)
    
    return goals

@app.get("/goals/{goal_id}", response_model=GoalResponse)
@limiter.limit("60/minute")
async def get_goal(
    request: Request,
    goal_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        goal = await db.goals.find_one({
            "_id": ObjectId(goal_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid goal ID format")
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    return goal

@app.put("/goals/{goal_id}", response_model=GoalResponse)
@limiter.limit("30/minute")
async def update_goal(
    request: Request,
    goal_id: str,
    goal_update: GoalUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if goal exists and belongs to user
        goal = await db.goals.find_one({
            "_id": ObjectId(goal_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid goal ID format")
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in goal_update.dict().items() if v is not None}
    if "subject_id" in update_data and update_data["subject_id"]:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Handle completed logic
    if "completed" in update_data and update_data["completed"] and not goal["completed"]:
        # Mark as completed now if not already completed
        if "completed_at" not in update_data or update_data["completed_at"] is None:
            update_data["completed_at"] = datetime.utcnow()
    
    # Update goal
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.goals.update_one(
            {"_id": ObjectId(goal_id)},
            {"$set": update_data}
        )
    
    # Return updated goal
    updated_goal = await db.goals.find_one({"_id": ObjectId(goal_id)})
    return updated_goal

@app.delete("/goals/{goal_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_goal(
    request: Request,
    goal_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if goal exists and belongs to user
        goal = await db.goals.find_one({
            "_id": ObjectId(goal_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid goal ID format")
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Delete goal
    await db.goals.delete_one({"_id": ObjectId(goal_id)})
    return None

# Note endpoints
@app.post("/notes", response_model=NoteResponse, status_code=201)
@limiter.limit("30/minute")
async def create_note(
    request: Request,
    note: NoteCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate subject_id if provided
    if note.subject_id:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(note.subject_id),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    new_note = {
        **note.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.notes.insert_one(new_note)
    created_note = await db.notes.find_one({"_id": result.inserted_id})
    
    return created_note

@app.get("/notes", response_model=List[NoteResponse])
@limiter.limit("60/minute")
async def get_notes(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if subject_id:
        try:
            query["subject_id"] = ObjectId(subject_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    if tag:
        query["tags"] = tag
    
    if search:
        search_query = {"$regex": search, "$options": "i"}
        query["$or"] = [
            {"title": search_query},
            {"content": search_query}
        ]
    
    # Get notes with filters
    notes = await db.notes.find(query).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    return notes

@app.get("/notes/{note_id}", response_model=NoteResponse)
@limiter.limit("60/minute")
async def get_note(
    request: Request,
    note_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        note = await db.notes.find_one({
            "_id": ObjectId(note_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid note ID format")
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return note

@app.put("/notes/{note_id}", response_model=NoteResponse)
@limiter.limit("30/minute")
async def update_note(
    request: Request,
    note_id: str,
    note_update: NoteUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if note exists and belongs to user
        note = await db.notes.find_one({
            "_id": ObjectId(note_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid note ID format")
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Validate subject_id if provided
    update_data = {k: v for k, v in note_update.dict().items() if v is not None}
    if "subject_id" in update_data and update_data["subject_id"]:
        try:
            subject = await db.subjects.find_one({
                "_id": ObjectId(update_data["subject_id"]),
                "user_id": ObjectId(current_user["_id"])
            })
            if not subject:
                raise HTTPException(status_code=404, detail="Subject not found")
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid subject ID format")
    
    # Update note
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await db.notes.update_one(
            {"_id": ObjectId(note_id)},
            {"$set": update_data}
        )
    
    # Return updated note
    updated_note = await db.notes.find_one({"_id": ObjectId(note_id)})
    return updated_note

@app.delete("/notes/{note_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_note(
    request: Request,
    note_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if note exists and belongs to user
        note = await db.notes.find_one({
            "_id": ObjectId(note_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid note ID format")
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Delete note
    await db.notes.delete_one({"_id": ObjectId(note_id)})
    return None

# Statistics endpoints
@app.get("/statistics", response_model=StatisticsResponse)
@limiter.limit("60/minute")
async def get_statistics(
    request: Request,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    # Default to last 30 days if no dates provided
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    user_id = ObjectId(current_user["_id"])
    
    # Assignment statistics
    total_assignments = await db.assignments.count_documents({
        "user_id": user_id,
        "created_at": {"$gte": start_date, "$lte": end_date}
    })
    
    completed_assignments = await db.assignments.count_documents({
        "user_id": user_id,
        "status": "completed",
        "created_at": {"$gte": start_date, "$lte": end_date}
    })
    
    pending_assignments = total_assignments - completed_assignments
    
    # Upcoming events
    upcoming_events = await db.events.count_documents({
        "user_id": user_id,
        "start_time": {"$gte": datetime.utcnow()}
    })
    
    # Study hours
    study_sessions = await db.study_sessions.find({
        "user_id": user_id,
        "completed": True,
        "completed_at": {"$gte": start_date, "$lte": end_date}
    }).to_list(1000)
    
    study_hours = sum(session.get("actual_duration", 0) for session in study_sessions) / 60.0
    
    # Subject statistics
    subjects = await db.subjects.find({"user_id": user_id}).to_list(100)
    subject_stats = []
    
    for subject in subjects:
        subject_id = subject["_id"]
        
        # Assignments for this subject
        subject_assignments = await db.assignments.count_documents({
            "user_id": user_id,
            "subject_id": subject_id,
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        completed_subject_assignments = await db.assignments.count_documents({
            "user_id": user_id,
            "subject_id": subject_id,
            "status": "completed",
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        # Study time for this subject
        subject_sessions = await db.study_sessions.find({
            "user_id": user_id,
            "subject_id": subject_id,
            "completed": True,
            "completed_at": {"$gte": start_date, "$lte": end_date}
        }).to_list(1000)
        
        subject_study_hours = sum(session.get("actual_duration", 0) for session in subject_sessions) / 60.0
        
        # Completion percentage
        completion_percentage = 0
        if subject_assignments > 0:
            completion_percentage = (completed_subject_assignments / subject_assignments) * 100
        
        subject_stats.append({
            "subject_id": str(subject_id),
            "subject_name": subject["name"],
            "color": subject["color"],
            "total_assignments": subject_assignments,
            "completed_assignments": completed_subject_assignments,
            "study_hours": subject_study_hours,
            "completion_percentage": completion_percentage
        })
    
    return {
        "total_assignments": total_assignments,
        "completed_assignments": completed_assignments,
        "pending_assignments": pending_assignments,
        "upcoming_events": upcoming_events,
        "study_hours": study_hours,
        "subject_stats": subject_stats
    }

# Health check endpoint for self-ping
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow()}

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
