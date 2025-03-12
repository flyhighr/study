# main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request, Form, Query
from gridfs import GridFS
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from fastapi.responses import StreamingResponse, ORJSONResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, HTMLResponse
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
import uuid, random
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
import json
import pytz
from datetime import datetime, timezone
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

import secrets
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize rate limiter with in-memory storage for faster access
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    strategy="fixed-window"
)

# Initialize FastAPI app with ORJSONResponse for faster serialization
app = FastAPI(
    title="Student Dashboard API",
    description="Backend API for Student Dashboard Application",
    version="1.0.0",
    default_response_class=ORJSONResponse
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

# MongoDB Configuration with optimized connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:password@cluster.mongodb.net/studentdashboard?retryWrites=true&w=majority")
client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGO_URI,
    maxPoolSize=100,
    minPoolSize=10,
    maxIdleTimeMS=30000,
    connectTimeoutMS=5000,
    socketTimeoutMS=10000,
    serverSelectionTimeoutMS=5000,
    waitQueueTimeoutMS=5000,
    retryWrites=True,
    retryReads=True
)
db = client.studentdashboard

# Initialize GridFS for file storage
fs = AsyncIOMotorGridFSBucket(db)

# Caches for better performance
USER_CACHE = {}
USER_CACHE_TTL = 300  # 5 minutes
JWT_CACHE = {}
JWT_CACHE_TTL = 60  # 1 minute
TIMEZONE_CACHE = {}

# Create indexes for better performance
async def create_indexes():
    # Basic indexes
    await db.users.create_index([("email", ASCENDING)], unique=True)
    await db.assignments.create_index([("user_id", ASCENDING)])
    await db.events.create_index([("user_id", ASCENDING)])
    await db.study_sessions.create_index([("user_id", ASCENDING)])
    await db.materials.create_index([("user_id", ASCENDING)])
    await db.subjects.create_index([("user_id", ASCENDING)])
    await db.goals.create_index([("user_id", ASCENDING)])
    await db.notes.create_index([("user_id", ASCENDING)])
    
    # Compound indexes for common query patterns
    await db.assignments.create_index([("user_id", ASCENDING), ("status", ASCENDING)])
    await db.assignments.create_index([("user_id", ASCENDING), ("due_date", ASCENDING)])
    await db.events.create_index([("user_id", ASCENDING), ("start_time", ASCENDING)])
    await db.study_sessions.create_index([("user_id", ASCENDING), ("completed", ASCENDING)])
    await db.study_sessions.create_index([("user_id", ASCENDING), ("scheduled_date", ASCENDING)])
    await db.goals.create_index([("user_id", ASCENDING), ("completed", ASCENDING)])
    
    # Email verification and password reset indexes
    await db.email_verification.create_index([("token", ASCENDING)], unique=True)
    await db.email_verification.create_index([("email", ASCENDING)])
    await db.email_verification.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    await db.password_reset.create_index([("token", ASCENDING)], unique=True)
    await db.password_reset.create_index([("email", ASCENDING)])
    await db.password_reset.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    
    # Assignment sharing indexes
    await db.assignment_shares.create_index([("share_id", ASCENDING)], unique=True)
    await db.assignment_shares.create_index([("user_id", ASCENDING)])
    await db.assignment_shares.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    
    # Notification indexes
    await db.notifications.create_index([("user_id", ASCENDING)])
    await db.notifications.create_index([("created_at", ASCENDING)])
    await db.notifications.create_index([("read", ASCENDING)])
    
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

# Default timezone if none is set
DEFAULT_TIMEZONE = "UTC"

# List of valid timezones
VALID_TIMEZONES = pytz.all_timezones

# Self-ping mechanism to prevent Render from shutting down
async def ping_server():
    app_url = os.getenv("APP_URL", "https://study-o5hp.onrender.com")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{app_url}/health", timeout=5) as response:
                if response.status == 200:
                    pass  # Success, no need to log every ping
                else:
                    logger.warning(f"Self-ping returned status: {response.status}")
        except Exception as e:
            logger.error(f"Self-ping failed: {str(e)}")

async def start_ping_task():
    while True:
        await ping_server()
        await asyncio.sleep(60 * 14)  # Ping every 14 minutes (Render free tier sleeps after 15 min)

# Background task for checking due dates and sending notifications
async def check_upcoming_due_dates():
    while True:
        try:
            # Get all users
            users = await db.users.find({}, {"_id": 1, "email": 1, "name": 1, "timezone": 1}).to_list(1000)
            
            for user in users:
                user_id = user["_id"]
                user_timezone = user.get("timezone", DEFAULT_TIMEZONE)
                
                # Check assignments due in the next 24 hours
                now_utc = datetime.now(timezone.utc)
                tomorrow_utc = now_utc + timedelta(hours=24)
                
                # Find assignments due soon that haven't been notified yet
                assignments_due_soon = await db.assignments.find({
                    "user_id": user_id,
                    "due_date": {"$gt": now_utc, "$lt": tomorrow_utc},
                    "status": {"$ne": "completed"},
                    "notification_sent": {"$ne": True}
                }).to_list(100)
                
                for assignment in assignments_due_soon:
                    # Convert due date to user timezone for email
                    due_date_user_tz = convert_to_user_timezone(assignment["due_date"], user_timezone)
                    due_date_str = due_date_user_tz.strftime("%B %d, %Y at %I:%M %p")
                    
                    # Create notification
                    notification = {
                        "user_id": user_id,
                        "type": "assignment_due",
                        "title": "Assignment Due Soon",
                        "message": f"Your assignment '{assignment['title']}' is due on {due_date_str}.",
                        "reference_id": str(assignment["_id"]),
                        "read": False,
                        "created_at": now_utc
                    }
                    
                    await db.notifications.insert_one(notification)
                    
                    # Send email notification
                    email_content = f'''
                    <html>
                        <body>
                            <h2>Assignment Due Reminder</h2>
                            <p>Hello {user['name']},</p>
                            <p>This is a reminder that your assignment <strong>{assignment['title']}</strong> is due on {due_date_str}.</p>
                            <p>Log in to your dashboard to view more details.</p>
                        </body>
                    </html>
                    '''
                    
                    # Send email in background
                    asyncio.create_task(send_email(
                        to_email=user["email"],
                        subject="Assignment Due Soon - Student Dashboard",
                        html_content=email_content
                    ))
                    
                    # Mark as notified
                    await db.assignments.update_one(
                        {"_id": assignment["_id"]},
                        {"$set": {"notification_sent": True}}
                    )
                
                # Check events starting in the next 24 hours
                events_starting_soon = await db.events.find({
                    "user_id": user_id,
                    "start_time": {"$gt": now_utc, "$lt": tomorrow_utc},
                    "notification_sent": {"$ne": True}
                }).to_list(100)
                
                for event in events_starting_soon:
                    # Convert start time to user timezone for email
                    start_time_user_tz = convert_to_user_timezone(event["start_time"], user_timezone)
                    start_time_str = start_time_user_tz.strftime("%B %d, %Y at %I:%M %p")
                    
                    # Create notification
                    notification = {
                        "user_id": user_id,
                        "type": "event_starting",
                        "title": "Event Starting Soon",
                        "message": f"Your event '{event['title']}' is starting on {start_time_str}.",
                        "reference_id": str(event["_id"]),
                        "read": False,
                        "created_at": now_utc
                    }
                    
                    await db.notifications.insert_one(notification)
                    
                    # Send email notification
                    email_content = f'''
                    <html>
                        <body>
                            <h2>Event Reminder</h2>
                            <p>Hello {user['name']},</p>
                            <p>This is a reminder that your event <strong>{event['title']}</strong> is starting on {start_time_str}.</p>
                            <p>Log in to your dashboard to view more details.</p>
                        </body>
                    </html>
                    '''
                    
                    # Send email in background
                    asyncio.create_task(send_email(
                        to_email=user["email"],
                        subject="Event Starting Soon - Student Dashboard",
                        html_content=email_content
                    ))
                    
                    # Mark as notified
                    await db.events.update_one(
                        {"_id": event["_id"]},
                        {"$set": {"notification_sent": True}}
                    )
            
        except Exception as e:
            logger.error(f"Error checking due dates: {str(e)}")
        
        # Run every hour
        await asyncio.sleep(60 * 60)
from json import JSONEncoder
from bson import ObjectId
from fastapi_cache.coder import JsonCoder

# Updated custom JSON encoder class
class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
# Create a custom coder class
class CustomJsonCoder(JsonCoder):
    @classmethod
    def encode(cls, value):
        return json.dumps(value, cls=CustomJsonEncoder)

# Update your FastAPI cache initialization in the startup event
@app.on_event("startup")
async def startup_event():
    await create_indexes()
    # Initialize in-memory cache with custom coder
    FastAPICache.init(InMemoryBackend(), coder=CustomJsonCoder)
    # Start background tasks
    asyncio.create_task(start_ping_task())
    asyncio.create_task(check_upcoming_due_dates())
# Timezone conversion utilities with caching
def get_timezone(timezone_str):
    if timezone_str not in TIMEZONE_CACHE:
        TIMEZONE_CACHE[timezone_str] = pytz.timezone(timezone_str)
    return TIMEZONE_CACHE[timezone_str]

def convert_to_user_timezone(utc_datetime, user_timezone):
    """Convert UTC datetime to user's timezone"""
    if not utc_datetime:
        return None
    
    # Ensure datetime has UTC timezone
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    
    # Convert to user timezone
    try:
        user_tz = get_timezone(user_timezone)
        return utc_datetime.astimezone(user_tz)
    except:
        # Fallback to UTC if timezone is invalid
        return utc_datetime

def convert_to_utc(local_datetime, user_timezone):
    """Convert local datetime to UTC"""
    if not local_datetime:
        return None
    
    # If datetime already has timezone info, convert to UTC
    if local_datetime.tzinfo is not None:
        return local_datetime.astimezone(timezone.utc)
    
    # Otherwise, assume it's in user's timezone and convert to UTC
    try:
        user_tz = get_timezone(user_timezone)
        local_with_tz = user_tz.localize(local_datetime)
        return local_with_tz.astimezone(timezone.utc)
    except:
        # Fallback to assuming it's UTC already if timezone is invalid
        return local_datetime.replace(tzinfo=timezone.utc)

# Optimized date processing for output
def process_dates_for_output(data, user_timezone):
    """Process all date fields in data for output to user's timezone"""
    # For lists of items, process them in a more optimized way
    if isinstance(data, list) and data and isinstance(data[0], dict):
        date_fields = set()
        # Find date fields from first item
        for key, value in data[0].items():
            if isinstance(value, datetime):
                date_fields.add(key)
        
        # Process date fields for all items in batch
        for item in data:
            for field in date_fields:
                if field in item and item[field]:
                    item[field] = convert_to_user_timezone(item[field], user_timezone)
        return data
    
    # Process dictionaries
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = convert_to_user_timezone(value, user_timezone)
            elif isinstance(value, dict) or isinstance(value, list):
                data[key] = process_dates_for_output(value, user_timezone)
    
    # Process lists
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = process_dates_for_output(item, user_timezone)
    
    return data

def process_dates_for_input(data, user_timezone):
    """Process all date fields in data from user's timezone to UTC"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = convert_to_utc(value, user_timezone)
            elif isinstance(value, dict) or isinstance(value, list):
                data[key] = process_dates_for_input(value, user_timezone)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = process_dates_for_input(item, user_timezone)
    return data

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

class EmailVerify(BaseModel):
    email: EmailStr

class VerifyToken(BaseModel):
    token: str

class PasswordReset(BaseModel):
    email: EmailStr

class ResetPassword(BaseModel):
    token: str
    new_password: str
    
    @validator('new_password')
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

class AssignmentShareCreate(BaseModel):
    user_id: str

class AssignmentShareResponse(BaseModel):
    share_link: str
    expires_at: Optional[datetime] = None

# Function to generate random tokens
def generate_random_token(length=64):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

# Function to generate short random links
def generate_short_link(length=8):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

# Email sending function
async def send_email(to_email: str, subject: str, html_content: str):
    # Replace with your email configuration
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "your-email@gmail.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "your-app-password")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = subject
    
    # Add HTML content
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

# Models for notifications
class NotificationResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    type: str
    title: str
    message: str
    reference_id: Optional[str] = None
    read: bool = False
    created_at: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class NotificationUpdate(BaseModel):
    read: bool = True

# Update UserCreate model to include email verification fields
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    timezone: Optional[str] = DEFAULT_TIMEZONE
    
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
    
    @validator('timezone')
    def validate_timezone(cls, v):
        if v not in VALID_TIMEZONES:
            raise ValueError(f'Invalid timezone. Please use a valid timezone identifier.')
        return v

# Update UserResponse to include verification status
class UserResponse(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr
    name: str
    timezone: str = DEFAULT_TIMEZONE
    created_at: datetime
    is_verified: bool = False

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    timezone: Optional[str] = None
    
    @validator('timezone')
    def validate_timezone(cls, v):
        if v is not None and v not in VALID_TIMEZONES:
            raise ValueError(f'Invalid timezone. Please use a valid timezone identifier.')
        return v

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
    notification_sent: Optional[bool] = None

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
    notification_sent: Optional[bool] = None

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
    
    @validator('end_time')
    def validate_end_time(cls, v, values, **kwargs):
        if v is not None and 'start_time' in values and values['start_time'] is not None and v < values['start_time']:
            raise ValueError('End time must be after start time')
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
        # This validation will be handled with timezone context in the endpoint
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

class DailyStudyData(BaseModel):
    date: str
    hours: float

class GoalStats(BaseModel):
    total: int
    completed: int
    completion_rate: float

class StatisticsResponse(BaseModel):
    total_assignments: int
    completed_assignments: int
    pending_assignments: int
    upcoming_events: int
    study_hours: float
    subject_stats: List[Dict[str, Any]]
    daily_study_data: List[DailyStudyData]
    avg_session_duration: float
    goal_stats: GoalStats
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        
# Authentication Functions - reduced bcrypt rounds for better performance
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=10)  # Reduced from default 12 to 10 for better performance
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# Optimized get_current_user with JWT caching
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Check cache first
    if token in JWT_CACHE and JWT_CACHE[token]["expire_time"] > time.time():
        return JWT_CACHE[token]["user"]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        
        # Get user with projection to fetch only needed fields
        user = await db.users.find_one(
            {"_id": ObjectId(user_id)},
            {"email": 1, "name": 1, "timezone": 1, "is_verified": 1, "created_at": 1}
        )
        
        if user is None:
            raise credentials_exception
        
        # Check if user is verified
        if not user.get("is_verified", False):
            raise HTTPException(
                status_code=403,
                detail="Email not verified. Please verify your email before accessing this resource."
            )
        
        # Cache the result
        JWT_CACHE[token] = {
            "user": user,
            "expire_time": time.time() + JWT_CACHE_TTL
        }
        
        return user
    except jwt.PyJWTError:
        raise credentials_exception

# Get cached user function
async def get_cached_user(user_id):
    cache_key = str(user_id)
    if cache_key in USER_CACHE and USER_CACHE[cache_key]["expire_time"] > time.time():
        return USER_CACHE[cache_key]["data"]
    
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        USER_CACHE[cache_key] = {
            "data": user,
            "expire_time": time.time() + USER_CACHE_TTL
        }
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
    
    # Set default timezone if not provided
    if not user.timezone:
        user.timezone = DEFAULT_TIMEZONE
    
    # Create user document
    new_user = {
        "email": user.email,
        "password": hashed_password,
        "name": user.name,
        "timezone": user.timezone,
        "is_verified": False,  # User starts as unverified
        "created_at": datetime.now(timezone.utc)
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
    
    # Generate verification token
    verification_token = generate_random_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    
    # Store verification token
    await db.email_verification.insert_one({
        "email": user.email,
        "user_id": result.inserted_id,
        "token": verification_token,
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at
    })
    
    # Send verification email
    verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/verify-email?token={verification_token}"
    email_content = f'''
    <html>
        <body>
            <h2>Welcome to Student Dashboard!</h2>
            <p>Hello {user.name},</p>
            <p>Thank you for registering. Please click the link below to verify your email:</p>
            <p><a href="{verification_url}">Verify Email Address</a></p>
            <p>This link will expire in 24 hours.</p>
            <p>If you did not create an account, please ignore this email.</p>
        </body>
    </html>
    '''
    
    # Send email in background
    asyncio.create_task(send_email(
        to_email=user.email,
        subject="Verify Your Email - Student Dashboard",
        html_content=email_content
    ))
    
    # Return created user without password
    created_user = await db.users.find_one({"_id": result.inserted_id})
    created_user.pop("password")
    return created_user

# Email verification endpoint
@app.post("/verify-email", response_model=dict)
@limiter.limit("20/minute")
async def verify_email(request: Request, verification: VerifyToken):
    # Find verification token
    token_data = await db.email_verification.find_one({
        "token": verification.token,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    
    # Update user verification status
    result = await db.users.update_one(
        {"_id": ObjectId(token_data["user_id"])},
        {"$set": {"is_verified": True}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete used token
    await db.email_verification.delete_one({"token": verification.token})
    
    return {"message": "Email successfully verified. You can now log in."}

# Resend verification email endpoint
@app.post("/resend-verification", response_model=dict)
@limiter.limit("5/minute")
async def resend_verification(request: Request, email_data: EmailVerify):
    # Find user
    user = await db.users.find_one({"email": email_data.email})
    
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If your email exists in our system, a verification link has been sent."}
    
    # Check if already verified
    if user.get("is_verified", False):
        return {"message": "Your email is already verified. You can log in."}
    
    # Delete any existing verification tokens
    await db.email_verification.delete_many({"email": email_data.email})
    
    # Generate new verification token
    verification_token = generate_random_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    
    # Store verification token
    await db.email_verification.insert_one({
        "email": email_data.email,
        "user_id": user["_id"],
        "token": verification_token,
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at
    })
    
    # Send verification email
    verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/verify-email?token={verification_token}"
    email_content = f'''
    <html>
        <body>
            <h2>Student Dashboard Email Verification</h2>
            <p>Hello {user['name']},</p>
            <p>Please click the link below to verify your email:</p>
            <p><a href="{verification_url}">Verify Email Address</a></p>
            <p>This link will expire in 24 hours.</p>
            <p>If you did not create an account, please ignore this email.</p>
        </body>
    </html>
    '''
    
    # Send email in background
    asyncio.create_task(send_email(
        to_email=email_data.email,
        subject="Verify Your Email - Student Dashboard",
        html_content=email_content
    ))
    
    return {"message": "If your email exists in our system, a verification link has been sent."}

# Forgot password endpoint
@app.post("/forgot-password", response_model=dict)
@limiter.limit("5/minute")
async def forgot_password(request: Request, password_data: PasswordReset):
    # Find user
    user = await db.users.find_one({"email": password_data.email})
    
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If your email exists in our system, a password reset link has been sent."}
    
    # Delete any existing password reset tokens
    await db.password_reset.delete_many({"email": password_data.email})
    
    # Generate password reset token
    reset_token = generate_random_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    
    # Store reset token
    await db.password_reset.insert_one({
        "email": password_data.email,
        "user_id": user["_id"],
        "token": reset_token,
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at
    })
    
    # Send password reset email
    reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={reset_token}"
    email_content = f'''
    <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>Hello {user['name']},</p>
            <p>We received a request to reset your password. Please click the link below to reset it:</p>
            <p><a href="{reset_url}">Reset Your Password</a></p>
            <p>This link will expire in 1 hour.</p>
            <p>If you did not request a password reset, please ignore this email.</p>
        </body>
    </html>
    '''
    
    # Send email in background
    asyncio.create_task(send_email(
        to_email=password_data.email,
        subject="Password Reset - Student Dashboard",
        html_content=email_content
    ))
    
    return {"message": "If your email exists in our system, a password reset link has been sent."}

# Reset password endpoint
@app.post("/reset-password", response_model=dict)
@limiter.limit("5/minute")
async def reset_password(request: Request, reset_data: ResetPassword):
    # Find reset token
    token_data = await db.password_reset.find_one({
        "token": reset_data.token,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Hash new password
    hashed_password = hash_password(reset_data.new_password)
    
    # Update user's password
    result = await db.users.update_one(
        {"_id": ObjectId(token_data["user_id"])},
        {"$set": {"password": hashed_password}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete used token
    await db.password_reset.delete_one({"token": reset_data.token})
    
    return {"message": "Password successfully reset. You can now log in with your new password."}

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
    
    # Check if email is verified
    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your email for the verification link."
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Create shareable assignments link
@app.post("/assignments/share", response_model=AssignmentShareResponse)
@limiter.limit("20/minute")
async def create_assignment_share(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["_id"]
    
    # Generate a unique share ID (between 5-10 chars)
    share_id = generate_short_link(random.randint(5, 10))
    
    # Set expiration (default to 7 days)
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    # Store share info
    share_data = {
        "share_id": share_id,
        "user_id": ObjectId(user_id),
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at
    }
    
    await db.assignment_shares.insert_one(share_data)
    
    # Generate shareable link
    share_link = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/shared-assignments/{share_id}"
    
    return {
        "share_link": share_link,
        "expires_at": expires_at
    }

# Get shared assignments - optimized version
@app.get("/shared-assignments/{share_id}")
@limiter.limit("30/minute")
@cache(expire=300)  # Cache for 5 minutes
async def get_shared_assignments(request: Request, share_id: str):
    # Find share data
    share_data = await db.assignment_shares.find_one({
        "share_id": share_id,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not share_data:
        raise HTTPException(status_code=404, detail="Shared link not found or expired")
    
    # Get user with projection to retrieve only needed fields
    user = await db.users.find_one(
        {"_id": share_data["user_id"]},
        {"name": 1, "timezone": 1}
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_timezone = user.get("timezone", DEFAULT_TIMEZONE)
    
    # Get pending assignments for the user with projection
    assignments = await db.assignments.find({
        "user_id": share_data["user_id"],
        "status": {"$ne": "completed"},
        "due_date": {"$gt": datetime.now(timezone.utc)}
    }, {
        "title": 1, 
        "description": 1, 
        "due_date": 1, 
        "subject_id": 1,
        "priority": 1,
        "status": 1
    }).sort("due_date", 1).to_list(100)
    
    # Get user's subjects with projection
    subjects = await db.subjects.find({
        "user_id": share_data["user_id"]
    }, {
        "name": 1
    }).to_list(100)
    
    subject_map = {str(subject["_id"]): subject["name"] for subject in subjects}
    
    # Process assignments for response
    processed_assignments = []
    for assignment in assignments:
        # Convert ObjectId to string
        assignment["_id"] = str(assignment["_id"])
        if "subject_id" in assignment and assignment["subject_id"]:
            assignment["subject_id"] = str(assignment["subject_id"])
        
        # Convert dates to user timezone
        assignment = process_dates_for_output(assignment, user_timezone)
        
        # Add subject name
        subject_id = str(assignment.get("subject_id", ""))
        assignment["subject_name"] = subject_map.get(subject_id, "Unknown Subject")
        
        processed_assignments.append(assignment)
    
    return {
        "user_name": user["name"],
        "assignments": processed_assignments,
        "total_count": len(processed_assignments)
    }

# Notification endpoints
@app.get("/notifications", response_model=List[NotificationResponse])
@limiter.limit("60/minute")
async def get_notifications(
    request: Request,
    skip: int = 0,
    limit: int = 20,
    unread_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    # Build filter query
    query = {"user_id": ObjectId(current_user["_id"])}
    
    if unread_only:
        query["read"] = False
    
    # Get notifications with filters
    notifications = await db.notifications.find(query).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    notifications = process_dates_for_output(notifications, user_timezone)
    
    return notifications

@app.put("/notifications/{notification_id}", response_model=NotificationResponse)
@limiter.limit("60/minute")
async def mark_notification_read(
    request: Request,
    notification_id: str,
    notification_update: NotificationUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if notification exists and belongs to user
        notification = await db.notifications.find_one({
            "_id": ObjectId(notification_id),
            "user_id": ObjectId(current_user["_id"])
        })
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid notification ID format")
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    # Update notification
    update_data = {"read": notification_update.read}
    await db.notifications.update_one(
        {"_id": ObjectId(notification_id)},
        {"$set": update_data}
    )
    
    # Return updated notification
    updated_notification = await db.notifications.find_one({"_id": ObjectId(notification_id)})
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    updated_notification = process_dates_for_output(updated_notification, user_timezone)
    
    return updated_notification

@app.get("/notifications/unread-count")
@limiter.limit("60/minute")
async def get_unread_notifications_count(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    # Count unread notifications
    unread_count = await db.notifications.count_documents({
        "user_id": ObjectId(current_user["_id"]),
        "read": False
    })
    
    return {"unread_count": unread_count}

@app.post("/notifications/mark-all-read")
@limiter.limit("20/minute")
async def mark_all_notifications_read(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    # Mark all user's notifications as read
    result = await db.notifications.update_many(
        {
            "user_id": ObjectId(current_user["_id"]),
            "read": False
        },
        {"$set": {"read": True}}
    )
    
    return {"marked_count": result.modified_count}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    user_data = {**current_user}
    return user_data

@app.put("/users/me", response_model=UserResponse)
async def update_user(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    # Build update data from provided fields
    update_data = {k: v for k, v in user_update.dict().items() if v is not None}
    
    if not update_data:
        # No update needed if no fields provided
        return current_user
    
    # Check if email is being updated and is not already taken
    if "email" in update_data and update_data["email"] != current_user["email"]:
        existing_user = await db.users.find_one({"email": update_data["email"]})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already taken")
    
    # Update user
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": update_data}
    )
    
    # Return updated user
    updated_user = await db.users.find_one({"_id": ObjectId(current_user["_id"])})
    
    # Clear user from cache
    cache_key = str(current_user["_id"])
    if cache_key in USER_CACHE:
        del USER_CACHE[cache_key]
    
    return updated_user

@app.get("/timezones")
@cache(expire=86400)  # Cache for 24 hours
async def get_timezones():
    """Return a list of valid timezones"""
    return {"timezones": VALID_TIMEZONES}

# Subject endpoints
@app.post("/subjects", response_model=SubjectResponse, status_code=201)
@limiter.limit("30/minute")
async def create_subject(request: Request, subject: SubjectCreate, current_user: dict = Depends(get_current_user)):
    new_subject = {
        **subject.dict(),
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.subjects.insert_one(new_subject)
    created_subject = await db.subjects.find_one({"_id": result.inserted_id})
    
    return created_subject

@app.get("/subjects", response_model=List[SubjectResponse])
@limiter.limit("60/minute")
@cache(expire=300)  # Cache for 5 minutes
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
        update_data["updated_at"] = datetime.now(timezone.utc)
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
    
    # Convert due_date from user timezone to UTC
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    assignment_dict = assignment.dict()
    assignment_dict["due_date"] = convert_to_utc(assignment_dict["due_date"], user_timezone)
    
    new_assignment = {
        **assignment_dict,
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.now(timezone.utc),
        "notification_sent": False
    }
    
    result = await db.assignments.insert_one(new_assignment)
    created_assignment = await db.assignments.find_one({"_id": result.inserted_id})
    
    # Convert dates back to user timezone for response
    created_assignment = process_dates_for_output(created_assignment, user_timezone)
    
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
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
        # Convert due_before from user timezone to UTC
        due_before_utc = convert_to_utc(due_before, user_timezone)
        date_query["$lte"] = due_before_utc
    if due_after:
        # Convert due_after from user timezone to UTC
        due_after_utc = convert_to_utc(due_after, user_timezone)
        date_query["$gte"] = due_after_utc
    
    if date_query:
        query["due_date"] = date_query
    
    # Get assignments with filters
    assignments = await db.assignments.find(query).sort("due_date", 1).skip(skip).limit(limit).to_list(limit)
    
    # Convert dates to user timezone for response
    assignments = process_dates_for_output(assignments, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    assignment = process_dates_for_output(assignment, user_timezone)
    
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
    
    # Convert due_date from user timezone to UTC if provided
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    if "due_date" in update_data:
        update_data["due_date"] = convert_to_utc(update_data["due_date"], user_timezone)
    
    # If status is changed to completed, reset notification_sent
    if "status" in update_data and update_data["status"] == "completed":
        update_data["notification_sent"] = False
    # If due date is changed, reset notification_sent
    elif "due_date" in update_data:
        update_data["notification_sent"] = False
    
    # Update assignment
    if update_data:
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.assignments.update_one(
            {"_id": ObjectId(assignment_id)},
            {"$set": update_data}
        )
    
    # Return updated assignment
    updated_assignment = await db.assignments.find_one({"_id": ObjectId(assignment_id)})
    
    # Convert dates to user timezone for response
    updated_assignment = process_dates_for_output(updated_assignment, user_timezone)
    
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
    
    # Delete related notifications
    await db.notifications.delete_many({
        "reference_id": assignment_id,
        "type": "assignment_due"
    })
    
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
    
    # Convert dates from user timezone to UTC
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    event_dict = event.dict()
    event_dict["start_time"] = convert_to_utc(event_dict["start_time"], user_timezone)
    event_dict["end_time"] = convert_to_utc(event_dict["end_time"], user_timezone)
    
    new_event = {
        **event_dict,
        "user_id": ObjectId(current_user["_id"]),
        "created_at": datetime.now(timezone.utc),
        "notification_sent": False
    }
    
    result = await db.events.insert_one(new_event)
    created_event = await db.events.find_one({"_id": result.inserted_id})
    
    # Convert dates back to user timezone for response
    created_event = process_dates_for_output(created_event, user_timezone)
    
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
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
        # Convert start_after from user timezone to UTC
        start_after_utc = convert_to_utc(start_after, user_timezone)
        date_query["$gte"] = start_after_utc
    if start_before:
        # Convert start_before from user timezone to UTC
        start_before_utc = convert_to_utc(start_before, user_timezone)
        date_query["$lte"] = start_before_utc
    
    if date_query:
        query["start_time"] = date_query
    
    # Get events with filters
    events = await db.events.find(query).sort("start_time", 1).skip(skip).limit(limit).to_list(limit)
    
    # Convert dates to user timezone for response
    events = process_dates_for_output(events, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    event = process_dates_for_output(event, user_timezone)
    
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
    
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
    
    # Convert datetime fields from user timezone to UTC
    if "start_time" in update_data:
        update_data["start_time"] = convert_to_utc(update_data["start_time"], user_timezone)
        # Reset notification status if start time changes
        update_data["notification_sent"] = False
    if "end_time" in update_data:
        update_data["end_time"] = convert_to_utc(update_data["end_time"], user_timezone)
    
    # Validate start/end time logic if both are provided
    if "start_time" in update_data and "end_time" in update_data:
        if update_data["end_time"] < update_data["start_time"]:
            raise HTTPException(status_code=400, detail="End time must be after start time")
    elif "start_time" in update_data:
        # Only start_time is being updated, check against existing end_time
        end_time_utc = event["end_time"]
        if update_data["start_time"] > end_time_utc:
            raise HTTPException(status_code=400, detail="Start time must be before end time")
    elif "end_time" in update_data:
        # Only end_time is being updated, check against existing start_time
        start_time_utc = event["start_time"]
        if update_data["end_time"] < start_time_utc:
            raise HTTPException(status_code=400, detail="End time must be after start time")
    
    # Update event
    if update_data:
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.events.update_one(
            {"_id": ObjectId(event_id)},
            {"$set": update_data}
        )
    
    # Return updated event
    updated_event = await db.events.find_one({"_id": ObjectId(event_id)})
    
    # Convert dates to user timezone for response
    updated_event = process_dates_for_output(updated_event, user_timezone)
    
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
    
    # Delete related notifications
    await db.notifications.delete_many({
        "reference_id": event_id,
        "type": "event_starting"
    })
    
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
    
    # Convert scheduled_date from user timezone to UTC
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    session_dict = study_session.dict()
    session_dict["scheduled_date"] = convert_to_utc(session_dict["scheduled_date"], user_timezone)
    
    new_study_session = {
        **session_dict,
        "user_id": ObjectId(current_user["_id"]),
        "completed": False,
        "actual_duration": None,
        "completed_at": None,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.study_sessions.insert_one(new_study_session)
    created_study_session = await db.study_sessions.find_one({"_id": result.inserted_id})
    
    # Convert dates to user timezone for response
    created_study_session = process_dates_for_output(created_study_session, user_timezone)
    
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
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
        # Convert scheduled_after from user timezone to UTC
        scheduled_after_utc = convert_to_utc(scheduled_after, user_timezone)
        date_query["$gte"] = scheduled_after_utc
    if scheduled_before:
        # Convert scheduled_before from user timezone to UTC
        scheduled_before_utc = convert_to_utc(scheduled_before, user_timezone)
        date_query["$lte"] = scheduled_before_utc
    
    if date_query:
        query["scheduled_date"] = date_query
    
    # Get study sessions with filters
    study_sessions = await db.study_sessions.find(query).sort("scheduled_date", 1).skip(skip).limit(limit).to_list(limit)
    
    # Convert dates to user timezone for response
    study_sessions = process_dates_for_output(study_sessions, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    study_session = process_dates_for_output(study_session, user_timezone)
    
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
    
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
    
    # Convert datetime fields from user timezone to UTC
    if "scheduled_date" in update_data:
        update_data["scheduled_date"] = convert_to_utc(update_data["scheduled_date"], user_timezone)
    if "completed_at" in update_data and update_data["completed_at"]:
        update_data["completed_at"] = convert_to_utc(update_data["completed_at"], user_timezone)
    
    # Handle completed logic
    if "completed" in update_data and update_data["completed"] and not study_session["completed"]:
        # Mark as completed now if not already completed
        if "completed_at" not in update_data or update_data["completed_at"] is None:
            update_data["completed_at"] = datetime.now(timezone.utc)
    
    # Update study session
    if update_data:
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.study_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": update_data}
        )
    
    # Return updated study session
    updated_session = await db.study_sessions.find_one({"_id": ObjectId(session_id)})
    
    # Convert dates to user timezone for response
    updated_session = process_dates_for_output(updated_session, user_timezone)
    
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
    
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    
    # Validate file size
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
    
    # Determine file type category
    file_type = "document"
    if file_ext in [".jpg", ".jpeg", ".png", ".gif"]:
        file_type = "image"
    elif file_ext in [".xlsx", ".xls"]:
        file_type = "spreadsheet"
    elif file_ext in [".pdf"]:
        file_type = "pdf"
    
    # Store file in GridFS
    file_id = await fs.upload_from_stream(
        unique_filename,
        io.BytesIO(file_content),
        metadata={
            "content_type": mimetypes.guess_type(file.filename)[0] or "application/octet-stream",
            "user_id": str(current_user["_id"]),
            "original_filename": file.filename
        }
    )
    
    # Create material record
    new_material = {
        "name": name,
        "description": description,
        "file_type": file_type,
        "file_size": file_size,
        "file_path": str(file_id),  # Store GridFS file ID as the path
        "original_filename": file.filename,  # Store original filename
        "subject_id": ObjectId(subject_id),
        "user_id": ObjectId(current_user["_id"]),
        "uploaded_at": datetime.now(timezone.utc)
    }
    
    result = await db.materials.insert_one(new_material)
    created_material = await db.materials.find_one({"_id": result.inserted_id})
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    created_material = process_dates_for_output(created_material, user_timezone)
    
    return created_material

@app.get("/materials/download/{material_id}")
@limiter.limit("20/minute")
async def download_material(
    request: Request,
    material_id: str,
    token: Optional[str] = Query(None),
    current_user: Optional[dict] = None
):
    # Allow authentication via token parameter or regular auth
    if token and not current_user:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("user_id")
            if user_id:
                # Use cache if available
                current_user = await get_cached_user(user_id)
        except:
            pass
    
    # If still no current_user, use the regular auth dependency
    if not current_user:
        current_user = await get_current_user(Depends(oauth2_scheme))
    
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
    
    # Get file ID from file_path
    try:
        file_id = ObjectId(material["file_path"])
    except InvalidId:
        raise HTTPException(status_code=500, detail="Invalid file reference")
    
    try:
        # Create a GridOut object to get the file
        grid_out = await fs.open_download_stream(file_id)
        
        # Get file data
        chunks = []
        async for chunk in grid_out:
            chunks.append(chunk)
        
        # Combine chunks
        file_data = b''.join(chunks)
        
        # Get content type from metadata or guess from filename
        content_type = grid_out.metadata.get("content_type", "application/octet-stream")
        
        # Get original filename or use material name with extension
        original_filename = material.get("original_filename") or f"{material['name']}"
        if not os.path.splitext(original_filename)[1]:
            # Add extension if missing
            if material["file_type"] == "pdf":
                original_filename += ".pdf"
            elif material["file_type"] == "document":
                original_filename += ".docx"
            elif material["file_type"] == "spreadsheet":
                original_filename += ".xlsx"
            elif material["file_type"] == "image":
                original_filename += ".jpg"
        
        # Return file as streaming response
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{original_filename}\""
            }
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving file")

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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    materials = process_dates_for_output(materials, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    material = process_dates_for_output(material, user_timezone)
    
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
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.materials.update_one(
            {"_id": ObjectId(material_id)},
            {"$set": update_data}
        )
    
    # Return updated material
    updated_material = await db.materials.find_one({"_id": ObjectId(material_id)})
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    updated_material = process_dates_for_output(updated_material, user_timezone)
    
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
    
    # Delete the file from GridFS
    try:
        file_id = ObjectId(material["file_path"])
        await fs.delete(file_id)
    except Exception as e:
        logger.error(f"Error deleting file from GridFS: {str(e)}")
    
    # Delete material record
    await db.materials.delete_one({"_id": ObjectId(material_id)})
    
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
    
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
    # Validate target date is in the future (using user's timezone)
    now_in_user_tz = convert_to_user_timezone(datetime.now(timezone.utc), user_timezone)
    target_date_in_user_tz = goal.target_date
    
    if target_date_in_user_tz.replace(tzinfo=None) < now_in_user_tz.replace(tzinfo=None):
        raise HTTPException(status_code=400, detail="Target date must be in the future")
    
    # Convert target_date from user timezone to UTC
    goal_dict = goal.dict()
    goal_dict["target_date"] = convert_to_utc(goal_dict["target_date"], user_timezone)
    
    new_goal = {
        **goal_dict,
        "user_id": ObjectId(current_user["_id"]),
        "progress": 0,
        "completed": False,
        "completed_at": None,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.goals.insert_one(new_goal)
    created_goal = await db.goals.find_one({"_id": result.inserted_id})
    
    # Convert dates back to user timezone for response
    created_goal = process_dates_for_output(created_goal, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    goals = process_dates_for_output(goals, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    goal = process_dates_for_output(goal, user_timezone)
    
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
    
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
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
    
    # Convert datetime fields from user timezone to UTC
    if "target_date" in update_data:
        # Validate target date is in the future (in user's timezone)
        now_in_user_tz = convert_to_user_timezone(datetime.now(timezone.utc), user_timezone)
        if update_data["target_date"].replace(tzinfo=None) < now_in_user_tz.replace(tzinfo=None):
            raise HTTPException(status_code=400, detail="Target date must be in the future")
        
        update_data["target_date"] = convert_to_utc(update_data["target_date"], user_timezone)
    
    if "completed_at" in update_data and update_data["completed_at"]:
        update_data["completed_at"] = convert_to_utc(update_data["completed_at"], user_timezone)
    
    # Handle completed logic
    if "completed" in update_data and update_data["completed"] and not goal["completed"]:
        # Mark as completed now if not already completed
        if "completed_at" not in update_data or update_data["completed_at"] is None:
            update_data["completed_at"] = datetime.now(timezone.utc)
    
    # Update goal
    if update_data:
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.goals.update_one(
            {"_id": ObjectId(goal_id)},
            {"$set": update_data}
        )
    
    # Return updated goal
    updated_goal = await db.goals.find_one({"_id": ObjectId(goal_id)})
    
    # Convert dates to user timezone for response
    updated_goal = process_dates_for_output(updated_goal, user_timezone)
    
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
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.notes.insert_one(new_note)
    created_note = await db.notes.find_one({"_id": result.inserted_id})
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    created_note = process_dates_for_output(created_note, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    notes = process_dates_for_output(notes, user_timezone)
    
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
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    note = process_dates_for_output(note, user_timezone)
    
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
        update_data["updated_at"] = datetime.now(timezone.utc)
        await db.notes.update_one(
            {"_id": ObjectId(note_id)},
            {"$set": update_data}
        )
    
    # Return updated note
    updated_note = await db.notes.find_one({"_id": ObjectId(note_id)})
    
    # Convert dates to user timezone for response
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    updated_note = process_dates_for_output(updated_note, user_timezone)
    
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
@cache(expire=60)  # Cache for 1 minute
async def get_statistics(
    request: Request,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    # Get user timezone
    user_timezone = current_user.get("timezone", DEFAULT_TIMEZONE)
    
    # Default to last 30 days if no dates provided
    now = datetime.now(timezone.utc)
    if not end_date:
        end_date = now
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Convert dates from user timezone to UTC
    start_date_utc = convert_to_utc(start_date, user_timezone)
    end_date_utc = convert_to_utc(end_date, user_timezone)
    
    user_id = ObjectId(current_user["_id"])
    
    # Assignment statistics - use projection to fetch only needed fields
    total_assignments = await db.assignments.count_documents({
        "user_id": user_id,
        "created_at": {"$gte": start_date_utc, "$lte": end_date_utc}
    })
    
    completed_assignments = await db.assignments.count_documents({
        "user_id": user_id,
        "status": "completed",
        "created_at": {"$gte": start_date_utc, "$lte": end_date_utc}
    })
    
    pending_assignments = total_assignments - completed_assignments
    
    # Upcoming events
    upcoming_events = await db.events.count_documents({
        "user_id": user_id,
        "start_time": {"$gte": now}
    })
    
    # Study hours - use projection to fetch only needed fields
    study_sessions = await db.study_sessions.find({
        "user_id": user_id,
        "completed": True,
        "completed_at": {"$gte": start_date_utc, "$lte": end_date_utc}
    }, {
        "actual_duration": 1
    }).to_list(1000)
    
    study_hours = sum(session.get("actual_duration", 0) for session in study_sessions) / 60.0
    
    # Daily study data for last 7 days
    daily_study_data = []
    today_utc = now
    
    for i in range(7):
        day_date = today_utc - timedelta(days=6-i)
        # Convert to user timezone for day boundaries
        day_date_user_tz = convert_to_user_timezone(day_date, user_timezone)
        day_start_user_tz = datetime(day_date_user_tz.year, day_date_user_tz.month, day_date_user_tz.day, 0, 0, 0)
        day_end_user_tz = datetime(day_date_user_tz.year, day_date_user_tz.month, day_date_user_tz.day, 23, 59, 59)
        
        # Convert back to UTC for query
        day_start_utc = convert_to_utc(day_start_user_tz, user_timezone)
        day_end_utc = convert_to_utc(day_end_user_tz, user_timezone)
        
        day_sessions = await db.study_sessions.find({
            "user_id": user_id,
            "completed": True,
            "completed_at": {"$gte": day_start_utc, "$lte": day_end_utc}
        }, {
            "actual_duration": 1
        }).to_list(100)
        
        day_minutes = sum(session.get("actual_duration", 0) for session in day_sessions)
        
        daily_study_data.append({
            "date": day_date_user_tz.strftime("%a"),
            "hours": day_minutes / 60.0
        })
    
    # Average session duration
    if study_sessions:
        avg_session_duration = sum(session.get("actual_duration", 0) for session in study_sessions) / len(study_sessions)
    else:
        avg_session_duration = 0
    
    # Subject statistics - use projection to fetch only needed fields
    subjects = await db.subjects.find({"user_id": user_id}, {
        "name": 1, "color": 1
    }).to_list(100)
    subject_stats = []
    
    for subject in subjects:
        subject_id = subject["_id"]
        
        # Assignments for this subject
        subject_assignments = await db.assignments.count_documents({
            "user_id": user_id,
            "subject_id": subject_id,
            "created_at": {"$gte": start_date_utc, "$lte": end_date_utc}
        })
        
        completed_subject_assignments = await db.assignments.count_documents({
            "user_id": user_id,
            "subject_id": subject_id,
            "status": "completed",
            "created_at": {"$gte": start_date_utc, "$lte": end_date_utc}
        })
        
        # Study time for this subject
        subject_sessions = await db.study_sessions.find({
            "user_id": user_id,
            "subject_id": subject_id,
            "completed": True,
            "completed_at": {"$gte": start_date_utc, "$lte": end_date_utc}
        }, {
            "actual_duration": 1
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
    
    # Goal statistics
    goals = await db.goals.find({
        "user_id": user_id,
        "created_at": {"$gte": start_date_utc, "$lte": end_date_utc}
    }, {
        "completed": 1
    }).to_list(100)
    
    total_goals = len(goals)
    completed_goals = sum(1 for goal in goals if goal.get("completed", False))
    goal_completion_rate = (completed_goals / total_goals * 100) if total_goals > 0 else 0
    
    return {
        "total_assignments": total_assignments,
        "completed_assignments": completed_assignments,
        "pending_assignments": pending_assignments,
        "upcoming_events": upcoming_events,
        "study_hours": study_hours,
        "subject_stats": subject_stats,
        "daily_study_data": daily_study_data,
        "avg_session_duration": avg_session_duration,
        "goal_stats": {
            "total": total_goals,
            "completed": completed_goals,
            "completion_rate": goal_completion_rate
        }
    }

@app.post("/export/pdf")
@limiter.limit("5/minute")
async def export_pdf(
    request: Request,
    data: str = Form(...),
    token: str = Form(...)
):
    try:
        # Validate token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user with projection
        user = await db.users.find_one(
            {"_id": ObjectId(user_id)},
            {"name": 1, "email": 1}
        )
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Parse data
        export_data = json.loads(data)
        
        # Generate PDF
        # This would typically use a library like ReportLab, PyPDF2, or WeasyPrint
        # For this example, we'll create a simple HTML response that could be printed as PDF
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Study Dashboard Export</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #4a6cf7; }}
                h2 {{ color: #333; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; }}
                .date {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Study Dashboard Export</h1>
                <p class="date">Generated on {datetime.utcnow().strftime("%B %d, %Y %H:%M")}</p>
            </div>
            <p>User: {user.get('name')} ({user.get('email')})</p>
        """
        
        # Add assignments if present
        if 'assignments' in export_data and export_data['assignments']:
            html_content += f"""
            <h2>Assignments ({len(export_data['assignments'])})</h2>
            <table>
                <tr>
                    <th>Title</th>
                    <th>Due Date</th>
                    <th>Priority</th>
                    <th>Status</th>
                </tr>
            """
            
            for assignment in export_data['assignments']:
                due_date = datetime.fromisoformat(assignment['due_date'].replace('Z', '+00:00'))
                html_content += f"""
                <tr>
                    <td>{assignment['title']}</td>
                    <td>{due_date.strftime("%b %d, %Y")}</td>
                    <td>{assignment['priority']}</td>
                    <td>{assignment['status']}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add events if present
        if 'events' in export_data and export_data['events']:
            html_content += f"""
            <h2>Events ({len(export_data['events'])})</h2>
            <table>
                <tr>
                    <th>Title</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Type</th>
                </tr>
            """
            
            for event in export_data['events']:
                start_time = datetime.fromisoformat(event['start_time'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(event['end_time'].replace('Z', '+00:00'))
                html_content += f"""
                <tr>
                    <td>{event['title']}</td>
                    <td>{start_time.strftime("%b %d, %Y %H:%M")}</td>
                    <td>{end_time.strftime("%b %d, %Y %H:%M")}</td>
                    <td>{event['type']}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add study sessions if present
        if 'study_sessions' in export_data and export_data['study_sessions']:
            html_content += f"""
            <h2>Study Sessions ({len(export_data['study_sessions'])})</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Planned Duration</th>
                    <th>Actual Duration</th>
                    <th>Completed</th>
                </tr>
            """
            
            for session in export_data['study_sessions']:
                scheduled_date = datetime.fromisoformat(session['scheduled_date'].replace('Z', '+00:00'))
                completed = "Yes" if session.get('completed', False) else "No"
                actual_duration = f"{session.get('actual_duration', 0)} min" if session.get('completed', False) else "-"
                
                html_content += f"""
                <tr>
                    <td>{scheduled_date.strftime("%b %d, %Y")}</td>
                    <td>{session['planned_duration']} min</td>
                    <td>{actual_duration}</td>
                    <td>{completed}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add goals if present
        if 'goals' in export_data and export_data['goals']:
            html_content += f"""
            <h2>Goals ({len(export_data['goals'])})</h2>
            <table>
                <tr>
                    <th>Title</th>
                    <th>Target Date</th>
                    <th>Progress</th>
                    <th>Completed</th>
                </tr>
            """
            
            for goal in export_data['goals']:
                target_date = datetime.fromisoformat(goal['target_date'].replace('Z', '+00:00'))
                completed = "Yes" if goal.get('completed', False) else "No"
                
                html_content += f"""
                <tr>
                    <td>{goal['title']}</td>
                    <td>{target_date.strftime("%b %d, %Y")}</td>
                    <td>{goal.get('progress', 0)}%</td>
                    <td>{completed}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating PDF export: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate export")
        
# Health check endpoint for self-ping
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
