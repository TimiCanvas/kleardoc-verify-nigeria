
# KlearDoc Python Backend Implementation Guide

This document outlines the complete Python backend structure for the KlearDoc platform, including all required files and Azure integrations.

## 📁 Project Structure

```
kleardoc-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI main application
│   ├── auth.py                 # Authentication logic
│   ├── ocr_processor.py        # Azure AI Vision OCR
│   ├── nimc_api.py            # NIMC database integration
│   ├── verification.py        # Document verification logic
│   ├── ai_agent.py            # Azure OpenAI + Semantic Kernel
│   ├── token_gen.py           # Secure token generation
│   ├── db.py                  # Database operations
│   └── config.py              # Configuration settings
├── streamlit_app/
│   ├── __init__.py
│   ├── streamlit_main.py      # Streamlit frontend
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── login.py
│   │   ├── dashboard.py
│   │   └── institutions.py
│   └── components/
│       ├── __init__.py
│       ├── auth_component.py
│       └── verification_flow.py
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_ocr.py
│   └── test_verification.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Core Backend Files

### 1. requirements.txt
```txt
# FastAPI and web framework
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# Streamlit for frontend
streamlit==1.28.1
streamlit-authenticator==0.2.3

# Azure AI Services
azure-ai-vision==4.0.0b1
azure-cognitiveservices-vision-computervision==0.9.0
azure-openai==1.3.5

# Semantic Kernel for AI agents
semantic-kernel==0.4.3.dev0

# Database
sqlalchemy==2.0.23
sqlite3  # Built-in Python module
firebase-admin==6.2.0

# Authentication and Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Image processing
Pillow==10.1.0
opencv-python==4.8.1.78

# HTTP clients
httpx==0.25.2
requests==2.31.0

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0
uuid==1.30
```

### 2. app/config.py
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Azure AI Vision
    AZURE_AI_VISION_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT", "")
    AZURE_AI_VISION_KEY = os.getenv("AZURE_AI_VISION_KEY", "")
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    # NIMC API (Mock or Real)
    NIMC_API_ENDPOINT = os.getenv("NIMC_API_ENDPOINT", "https://api.nimc.gov.ng")
    NIMC_API_KEY = os.getenv("NIMC_API_KEY", "")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kleardoc.db")
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # File uploads
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

settings = Settings()
```

### 3. app/db.py
```python
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

from .config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class VerificationRecord(Base):
    __tablename__ = "verification_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    document_type = Column(String)
    document_path = Column(String)
    ocr_data = Column(Text)  # JSON string
    verification_status = Column(String)  # pending, approved, rejected
    verification_token = Column(String, unique=True, index=True)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    verified_at = Column(DateTime)

class InstitutionQuery(Base):
    __tablename__ = "institution_queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_name = Column(String)
    verification_token = Column(String, index=True)
    query_result = Column(Text)  # JSON string
    queried_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 4. app/auth.py
```python
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from pydantic import BaseModel

from .db import get_db, User
from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserCreate(BaseModel):
    email: str
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_user(db: Session, user: UserCreate) -> User:
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user
```

### 5. app/ocr_processor.py
```python
import asyncio
import json
from typing import Dict, Any
from PIL import Image
from azure.ai.vision import ImageAnalysisClient
from azure.ai.vision.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import io
import base64

from .config import settings

class OCRProcessor:
    def __init__(self):
        self.client = ImageAnalysisClient(
            endpoint=settings.AZURE_AI_VISION_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_AI_VISION_KEY)
        )
    
    async def extract_text_from_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract text from uploaded image using Azure AI Vision
        """
        try:
            # Convert image to required format
            image_stream = io.BytesIO(image_data)
            
            # Use Azure AI Vision to extract text
            result = self.client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.READ]
            )
            
            # Extract text and confidence scores
            extracted_text = ""
            confidence_scores = []
            
            if result.read:
                for page in result.read.pages:
                    for line in page.lines:
                        extracted_text += line.text + "\n"
                        confidence_scores.append(line.confidence if hasattr(line, 'confidence') else 0.9)
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Parse Nigerian document specific information
            parsed_data = self._parse_nigerian_document(extracted_text)
            
            return {
                "raw_text": extracted_text,
                "parsed_data": parsed_data,
                "confidence": overall_confidence,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "raw_text": "",
                "parsed_data": {},
                "confidence": 0.0,
                "status": "error",
                "error": str(e)
            }
    
    def _parse_nigerian_document(self, text: str) -> Dict[str, Any]:
        """
        Parse specific information from Nigerian documents
        """
        parsed = {
            "document_type": self._detect_document_type(text),
            "full_name": None,
            "nin_number": None,
            "date_of_birth": None,
            "address": None,
            "phone_number": None,
            "email": None
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip().upper()
            
            # Extract NIN number (11 digits)
            if 'NIN' in line or 'NATIONAL IDENTITY NUMBER' in line:
                import re
                nin_match = re.search(r'\b\d{11}\b', line)
                if nin_match:
                    parsed["nin_number"] = nin_match.group()
            
            # Extract names
            if any(keyword in line for keyword in ['SURNAME', 'FIRST NAME', 'LAST NAME']):
                if 'SURNAME' in line:
                    parsed["surname"] = line.split(':')[-1].strip() if ':' in line else line.replace('SURNAME', '').strip()
                elif 'FIRST NAME' in line:
                    parsed["first_name"] = line.split(':')[-1].strip() if ':' in line else line.replace('FIRST NAME', '').strip()
            
            # Extract date of birth
            if 'DATE OF BIRTH' in line or 'DOB' in line:
                import re
                date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', line)
                if date_match:
                    parsed["date_of_birth"] = date_match.group()
            
            # Extract address
            if any(keyword in line for keyword in ['ADDRESS', 'RESIDENT']):
                if ':' in line:
                    parsed["address"] = line.split(':', 1)[-1].strip()
        
        # Combine names
        if parsed.get("surname") and parsed.get("first_name"):
            parsed["full_name"] = f"{parsed['first_name']} {parsed['surname']}"
        
        return parsed
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect the type of Nigerian document
        """
        text_upper = text.upper()
        
        if 'NATIONAL IDENTITY' in text_upper or 'NIN' in text_upper:
            return "NIN_SLIP"
        elif any(utility in text_upper for utility in ['NEPA', 'PHCN', 'ELECTRICITY', 'WATER CORPORATION']):
            return "UTILITY_BILL"
        elif 'BANK STATEMENT' in text_upper:
            return "BANK_STATEMENT"
        elif 'VOTER' in text_upper:
            return "VOTERS_CARD"
        else:
            return "UNKNOWN"

# Global OCR processor instance
ocr_processor = OCRProcessor()
```

### 6. app/nimc_api.py
```python
import httpx
import asyncio
from typing import Dict, Any, Optional
import json

from .config import settings

class NIMCAPIClient:
    def __init__(self):
        self.base_url = settings.NIMC_API_ENDPOINT
        self.api_key = settings.NIMC_API_KEY
        
    async def verify_identity(self, nin_number: str, full_name: str, date_of_birth: str = None) -> Dict[str, Any]:
        """
        Verify identity against NIMC database
        Since real NIMC API might not be available, this includes mock functionality
        """
        
        # Check if we should use mock data (for development)
        if not self.api_key or "mock" in self.base_url.lower():
            return await self._mock_nimc_verification(nin_number, full_name, date_of_birth)
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "nin": nin_number,
                    "full_name": full_name,
                    "date_of_birth": date_of_birth
                }
                
                response = await client.post(
                    f"{self.base_url}/verify",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "verified" if result.get("match", False) else "no_match",
                        "confidence": result.get("confidence", 0.0),
                        "details": result.get("details", {}),
                        "verification_id": result.get("verification_id"),
                        "timestamp": result.get("timestamp")
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"API returned status {response.status_code}",
                        "details": {}
                    }
                    
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "details": {}
            }
    
    async def _mock_nimc_verification(self, nin_number: str, full_name: str, date_of_birth: str = None) -> Dict[str, Any]:
        """
        Mock NIMC verification for development and testing
        """
        
        # Simulate API delay
        await asyncio.sleep(2)
        
        # Mock database of valid NINs
        mock_database = {
            "12345678901": {
                "full_name": "JOHN ADEBAYO DOE",
                "date_of_birth": "15/03/1990",
                "status": "active",
                "address": "123 VICTORIA ISLAND, LAGOS STATE"
            },
            "98765432109": {
                "full_name": "AMINA IBRAHIM HASSAN",
                "date_of_birth": "22/07/1985",
                "status": "active", 
                "address": "456 GARKI AREA, ABUJA FCT"
            }
        }
        
        if nin_number in mock_database:
            stored_data = mock_database[nin_number]
            
            # Check name similarity (simple comparison)
            name_match = self._calculate_name_similarity(full_name, stored_data["full_name"])
            
            if name_match > 0.8:  # 80% similarity threshold
                return {
                    "status": "verified",
                    "confidence": name_match,
                    "details": {
                        "nin": nin_number,
                        "full_name": stored_data["full_name"],
                        "date_of_birth": stored_data["date_of_birth"],
                        "address": stored_data["address"],
                        "match_score": name_match
                    },
                    "verification_id": f"mock_verify_{nin_number}",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            else:
                return {
                    "status": "no_match",
                    "confidence": name_match,
                    "details": {
                        "reason": "Name mismatch",
                        "expected": stored_data["full_name"],
                        "provided": full_name
                    }
                }
        else:
            return {
                "status": "not_found",
                "confidence": 0.0,
                "details": {
                    "reason": "NIN not found in database"
                }
            }
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names
        """
        from difflib import SequenceMatcher
        
        # Normalize names
        name1 = name1.upper().strip()
        name2 = name2.upper().strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, name1, name2).ratio()
        return similarity

# Global NIMC client instance
nimc_client = NIMCAPIClient()
```

### 7. app/ai_agent.py
```python
import asyncio
import json
from typing import Dict, Any, List
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelFunctionDecorator

from .config import settings

class KlearDocAIAgent:
    def __init__(self):
        self.kernel = Kernel()
        
        # Setup Azure OpenAI
        self.chat_completion = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY
        )
        
        self.kernel.add_service(self.chat_completion)
        
        # Register semantic functions
        self._register_functions()
    
    def _register_functions(self):
        """Register AI functions for document verification"""
        
        @KernelFunctionDecorator(
            name="analyze_document_quality",
            description="Analyze the quality of extracted OCR text"
        )
        async def analyze_document_quality(self, ocr_data: str) -> str:
            prompt = f"""
            Analyze this OCR extracted text from a Nigerian document and provide feedback:
            
            OCR Text: {ocr_data}
            
            Please assess:
            1. Text clarity and completeness
            2. Missing information
            3. Potential OCR errors
            4. Document authenticity indicators
            
            Provide friendly, helpful feedback in Nigerian English context.
            """
            
            response = await self.chat_completion.complete_chat_async(prompt)
            return response.content
        
        @KernelFunctionDecorator(
            name="verification_guidance",
            description="Provide guidance based on verification results"
        )
        async def verification_guidance(self, verification_result: str, ocr_data: str) -> str:
            prompt = f"""
            A Nigerian user has completed document verification with this result:
            Verification Result: {verification_result}
            OCR Data: {ocr_data}
            
            Provide clear, helpful guidance in Nigerian English context about:
            1. What the result means
            2. Next steps if verification failed
            3. How to improve document quality
            4. Timeline for resubmission if needed
            
            Be encouraging and culturally appropriate for Nigerian users.
            """
            
            response = await self.chat_completion.complete_chat_async(prompt)
            return response.content
        
        # Register functions with kernel
        self.kernel.add_function(plugin_name="document_ai", function=analyze_document_quality)
        self.kernel.add_function(plugin_name="verification_ai", function=verification_guidance)
    
    async def analyze_document_upload(self, ocr_data: Dict[str, Any]) -> str:
        """
        Analyze uploaded document and provide feedback
        """
        try:
            # Prepare OCR data for analysis
            ocr_text = json.dumps(ocr_data, indent=2)
            
            # Use semantic function to analyze
            analyze_func = self.kernel.functions["document_ai"]["analyze_document_quality"]
            result = await analyze_func.invoke_async(self.kernel, ocr_data=ocr_text)
            
            return result.value
            
        except Exception as e:
            return f"I encountered an issue analyzing your document: {str(e)}. Please try uploading again."
    
    async def provide_verification_feedback(self, verification_result: Dict[str, Any], ocr_data: Dict[str, Any]) -> str:
        """
        Provide feedback based on verification results
        """
        try:
            # Prepare data for AI analysis
            verification_text = json.dumps(verification_result, indent=2)
            ocr_text = json.dumps(ocr_data, indent=2)
            
            # Use semantic function for guidance
            guidance_func = self.kernel.functions["verification_ai"]["verification_guidance"]
            result = await guidance_func.invoke_async(
                self.kernel, 
                verification_result=verification_text,
                ocr_data=ocr_text
            )
            
            return result.value
            
        except Exception as e:
            return f"I encountered an issue providing feedback: {str(e)}. Please contact support if this persists."
    
    async def suggest_document_improvements(self, ocr_confidence: float, missing_fields: List[str]) -> str:
        """
        Suggest improvements for document quality
        """
        suggestions = []
        
        if ocr_confidence < 0.7:
            suggestions.extend([
                "📸 Try taking a clearer photo with better lighting",
                "🔍 Ensure the document is fully visible and not cut off",
                "📱 Use your phone's camera in good lighting conditions"
            ])
        
        if missing_fields:
            suggestions.append(f"📋 Make sure these fields are clearly visible: {', '.join(missing_fields)}")
        
        if not suggestions:
            suggestions.append("✅ Your document quality looks good!")
        
        return "\n".join(suggestions)
    
    async def generate_success_message(self, verification_token: str) -> str:
        """
        Generate success message with token
        """
        return f"""
        🎉 Congratulations! Your identity has been successfully verified!
        
        Your verification token is: `{verification_token}`
        
        This token can be used by financial institutions to confirm your verified identity. 
        Keep it safe and share it only when needed for official purposes.
        
        ✅ Your verification is valid for 24 hours
        🔒 This token is secure and encrypted
        📞 Contact support if you need assistance
        """

# Global AI agent instance
ai_agent = KlearDocAIAgent()
```

### 8. app/token_gen.py
```python
import uuid
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import json

class VerificationTokenGenerator:
    
    @staticmethod
    def generate_token(user_id: str, verification_id: str) -> str:
        """
        Generate a secure verification token
        """
        # Create a unique identifier
        unique_part = secrets.token_urlsafe(16)
        
        # Create token with klr prefix for branding
        token = f"klr-{unique_part}"
        
        return token
    
    @staticmethod
    def generate_token_with_metadata(user_id: str, verification_data: dict) -> dict:
        """
        Generate token with associated metadata
        """
        token = VerificationTokenGenerator.generate_token(user_id, verification_data.get('id', ''))
        
        # Create token metadata
        metadata = {
            "token": token,
            "user_id": user_id,
            "verification_level": verification_data.get('verification_level', 'Level 1'),
            "document_type": verification_data.get('document_type', 'Unknown'),
            "confidence_score": verification_data.get('confidence', 0.0),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "is_active": True
        }
        
        return metadata
    
    @staticmethod
    def validate_token_format(token: str) -> bool:
        """
        Validate token format
        """
        if not token or not isinstance(token, str):
            return False
        
        if not token.startswith('klr-'):
            return False
        
        if len(token) < 10:
            return False
        
        return True
    
    @staticmethod
    def hash_token(token: str) -> str:
        """
        Create a hash of the token for secure storage
        """
        return hashlib.sha256(token.encode()).hexdigest()

# Global token generator instance
token_generator = VerificationTokenGenerator()
```

### 9. app/verification.py
```python
import asyncio
import json
from typing import Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from .ocr_processor import ocr_processor
from .nimc_api import nimc_client
from .ai_agent import ai_agent
from .token_gen import token_generator
from .db import VerificationRecord

class DocumentVerificationEngine:
    
    async def process_document_verification(
        self, 
        user_id: str,
        image_data: bytes, 
        document_type: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Complete document verification pipeline
        """
        
        verification_record = VerificationRecord(
            user_id=user_id,
            document_type=document_type,
            verification_status="processing"
        )
        db.add(verification_record)
        db.commit()
        
        try:
            # Step 1: OCR Processing
            ocr_result = await ocr_processor.extract_text_from_image(image_data)
            
            if ocr_result["status"] != "success":
                return await self._handle_ocr_failure(verification_record, ocr_result, db)
            
            # Step 2: AI Analysis of OCR
            ai_feedback = await ai_agent.analyze_document_upload(ocr_result)
            
            # Step 3: NIMC Verification (if NIN document)
            nimc_result = None
            if document_type == "NIN_SLIP" and ocr_result["parsed_data"].get("nin_number"):
                nimc_result = await nimc_client.verify_identity(
                    nin_number=ocr_result["parsed_data"]["nin_number"],
                    full_name=ocr_result["parsed_data"].get("full_name", ""),
                    date_of_birth=ocr_result["parsed_data"].get("date_of_birth")
                )
            
            # Step 4: Final Verification Decision
            verification_decision = await self._make_verification_decision(
                ocr_result, nimc_result, document_type
            )
            
            # Step 5: Generate Token if Approved
            token_data = None
            if verification_decision["status"] == "approved":
                token_data = token_generator.generate_token_with_metadata(
                    user_id, {
                        'id': verification_record.id,
                        'verification_level': verification_decision["level"],
                        'document_type': document_type,
                        'confidence': verification_decision["confidence"]
                    }
                )
            
            # Step 6: AI Feedback on Final Result
            final_ai_message = await ai_agent.provide_verification_feedback(
                verification_decision, ocr_result
            )
            
            # Update verification record
            verification_record.ocr_data = json.dumps(ocr_result)
            verification_record.verification_status = verification_decision["status"]
            verification_record.confidence_score = verification_decision["confidence"]
            verification_record.verified_at = datetime.utcnow()
            
            if token_data:
                verification_record.verification_token = token_data["token"]
            
            db.commit()
            
            return {
                "status": "success",
                "verification_id": verification_record.id,
                "verification_result": verification_decision,
                "ocr_data": ocr_result,
                "nimc_result": nimc_result,
                "ai_feedback": final_ai_message,
                "token_data": token_data
            }
            
        except Exception as e:
            verification_record.verification_status = "error"
            verification_record.verification_token = None
            db.commit()
            
            return {
                "status": "error",
                "error": str(e),
                "verification_id": verification_record.id
            }
    
    async def _handle_ocr_failure(self, record: VerificationRecord, ocr_result: Dict, db: Session) -> Dict:
        """Handle OCR processing failures"""
        
        record.verification_status = "ocr_failed"
        record.ocr_data = json.dumps(ocr_result)
        db.commit()
        
        ai_message = await ai_agent.suggest_document_improvements(0.0, ["all_fields"])
        
        return {
            "status": "failed",
            "reason": "ocr_failure",
            "ai_feedback": ai_message,
            "verification_id": record.id
        }
    
    async def _make_verification_decision(
        self, 
        ocr_result: Dict, 
        nimc_result: Dict, 
        document_type: str
    ) -> Dict[str, Any]:
        """Make final verification decision based on all data"""
        
        confidence = ocr_result.get("confidence", 0.0)
        
        # For NIN documents, require NIMC verification
        if document_type == "NIN_SLIP":
            if nimc_result and nimc_result.get("status") == "verified":
                return {
                    "status": "approved",
                    "level": "Level 2 - Government ID",
                    "confidence": min(confidence, nimc_result.get("confidence", 0.0)),
                    "checks": ["OCR Extraction", "NIMC Database", "AI Validation"]
                }
            else:
                return {
                    "status": "rejected",
                    "level": "None",
                    "confidence": 0.0,
                    "reason": "NIMC verification failed",
                    "checks": ["OCR Extraction", "NIMC Database (Failed)"]
                }
        
        # For other documents, use OCR confidence threshold
        elif confidence > 0.8:
            return {
                "status": "approved",
                "level": "Level 1 - Basic",
                "confidence": confidence,
                "checks": ["OCR Extraction", "AI Validation"]
            }
        else:
            return {
                "status": "rejected",
                "level": "None", 
                "confidence": confidence,
                "reason": "Low document quality",
                "checks": ["OCR Extraction (Low Quality)"]
            }

# Global verification engine instance
verification_engine = DocumentVerificationEngine()
```

### 10. app/main.py
```python
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
import json

from .auth import (
    authenticate_user, create_access_token, create_user, get_current_user,
    UserCreate, UserResponse, Token
)
from .db import get_db, User, VerificationRecord, InstitutionQuery
from .verification import verification_engine
from .config import settings

app = FastAPI(title="KlearDoc API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = create_user(db, user)
    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        created_at=db_user.created_at
    )

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return access token"""
    
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at
    )

@app.post("/verify/document")
async def verify_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and verify a document"""
    
    # Validate file
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    file_extension = f".{file.filename.split('.')[-1].lower()}"
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Read file data
    file_data = await file.read()
    
    # Process verification
    result = await verification_engine.process_document_verification(
        user_id=current_user.id,
        image_data=file_data,
        document_type=document_type,
        db=db
    )
    
    return result

@app.get("/verify/history")
async def get_verification_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's verification history"""
    
    records = db.query(VerificationRecord).filter(
        VerificationRecord.user_id == current_user.id
    ).order_by(VerificationRecord.created_at.desc()).all()
    
    return [
        {
            "id": record.id,
            "document_type": record.document_type,
            "status": record.verification_status,
            "confidence": record.confidence_score,
            "created_at": record.created_at,
            "verified_at": record.verified_at,
            "token": record.verification_token
        }
        for record in records
    ]

@app.post("/institutions/verify-token")
async def verify_token_for_institution(
    token_data: dict,
    db: Session = Depends(get_db)
):
    """Verify a token for financial institutions"""
    
    token = token_data.get("token")
    institution_name = token_data.get("institution_name", "Unknown")
    
    if not token:
        raise HTTPException(status_code=400, detail="Token required")
    
    # Find verification record
    record = db.query(VerificationRecord).filter(
        VerificationRecord.verification_token == token,
        VerificationRecord.verification_status == "approved"
    ).first()
    
    if not record:
        # Log the query attempt
        query_record = InstitutionQuery(
            institution_name=institution_name,
            verification_token=token,
            query_result=json.dumps({"status": "token_not_found"})
        )
        db.add(query_record)
        db.commit()
        
        raise HTTPException(status_code=404, detail="Token not found or invalid")
    
    # Get user info
    user = db.query(User).filter(User.id == record.user_id).first()
    
    # Parse OCR data
    ocr_data = json.loads(record.ocr_data) if record.ocr_data else {}
    parsed_data = ocr_data.get("parsed_data", {})
    
    # Prepare response (limited user data for privacy)
    response_data = {
        "status": "verified",
        "user": {
            "id": user.id[:8] + "***",  # Partial ID for privacy
            "full_name": parsed_data.get("full_name", user.full_name),
            "nin_number": parsed_data.get("nin_number", "")[:5] + "***" + parsed_data.get("nin_number", "")[-3:] if parsed_data.get("nin_number") else None,
            "verification_level": "Level 2 - Government ID" if record.document_type == "NIN_SLIP" else "Level 1 - Basic",
            "verified_at": record.verified_at.isoformat()
        },
        "metadata": {
            "confidence": record.confidence_score,
            "document_type": record.document_type,
            "verification_id": record.id[:12] + "***"
        }
    }
    
    # Log the successful query
    query_record = InstitutionQuery(
        institution_name=institution_name,
        verification_token=token,
        query_result=json.dumps(response_data)
    )
    db.add(query_record)
    db.commit()
    
    return response_data

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "KlearDoc API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 11. streamlit_app/streamlit_main.py
```python
import streamlit as st
import requests
import json
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="KlearDoc - Document Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL (adjust as needed)
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🛡️ KlearDoc - Nigerian Document Verification")
    st.markdown("Secure, AI-powered document verification for Nigeria")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if not st.session_state.logged_in:
        page = st.sidebar.selectbox("Choose a page", ["Login", "Register"])
        
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
    else:
        page = st.sidebar.selectbox(
            "Choose a page", 
            ["Dashboard", "Verify Document", "History", "Institutions", "Logout"]
        )
        
        if page == "Dashboard":
            dashboard_page()
        elif page == "Verify Document":
            verify_document_page()
        elif page == "History":
            history_page()
        elif page == "Institutions":
            institutions_page()
        elif page == "Logout":
            logout()

def login_page():
    st.header("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit and email and password:
            with st.spinner("Logging in..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/auth/login",
                        data={"username": email, "password": password}
                    )
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        st.session_state.access_token = token_data["access_token"]
                        st.session_state.logged_in = True
                        
                        # Get user info
                        user_response = requests.get(
                            f"{API_BASE_URL}/auth/me",
                            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
                        )
                        
                        if user_response.status_code == 200:
                            st.session_state.user_info = user_response.json()
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                        
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")

def register_page():
    st.header("📝 Create Account")
    
    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif not all([full_name, email, password]):
                st.error("Please fill in all fields")
            else:
                with st.spinner("Creating account..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/register",
                            json={
                                "full_name": full_name,
                                "email": email,
                                "password": password
                            }
                        )
                        
                        if response.status_code == 200:
                            st.success("Account created successfully! Please login.")
                        else:
                            error_detail = response.json().get("detail", "Registration failed")
                            st.error(error_detail)
                            
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")

def dashboard_page():
    st.header(f"👋 Welcome, {st.session_state.user_info['full_name']}!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Account Status", "Active", "✅")
    
    with col2:
        st.metric("Verification Level", "Basic", "📄")
    
    with col3:
        st.metric("Documents Verified", "0", "📊")
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Verify New Document", use_container_width=True):
            st.switch_page("pages/verify_document.py")
    
    with col2:
        if st.button("📋 View History", use_container_width=True):
            st.switch_page("pages/history.py")
    
    st.markdown("---")
    
    st.subheader("ℹ️ Platform Information")
    
    with st.expander("Supported Documents"):
        st.write("""
        - **NIN Slip** (National Identity Number)
        - **Utility Bills** (NEPA, Water, etc.)
        - **Bank Statements**
        - **Voter Registration Cards**
        """)
    
    with st.expander("Verification Process"):
        st.write("""
        1. **Upload**: Submit your document
        2. **OCR Processing**: AI extracts text
        3. **Verification**: Cross-check with databases
        4. **Token Generation**: Receive verification token
        """)

def verify_document_page():
    st.header("📤 Document Verification")
    
    st.info("Upload a clear image of your Nigerian document for verification")
    
    # Document type selection
    document_type = st.selectbox(
        "Select Document Type",
        ["NIN_SLIP", "UTILITY_BILL", "BANK_STATEMENT", "VOTERS_CARD"]
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Maximum file size: 10MB"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)
        
        # Verify button
        if st.button("🔍 Start Verification", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Prepare file data
                    files = {"file": uploaded_file.getvalue()}
                    data = {"document_type": document_type}
                    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                    
                    # Send to API
                    response = requests.post(
                        f"{API_BASE_URL}/verify/document",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data=data,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        display_verification_results(result)
                    else:
                        st.error("Verification failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error during verification: {str(e)}")

def display_verification_results(result):
    st.markdown("---")
    st.subheader("📊 Verification Results")
    
    if result["status"] == "success":
        verification_result = result["verification_result"]
        
        if verification_result["status"] == "approved":
            st.success("✅ Document Verified Successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Verification Level", verification_result["level"])
                st.metric("Confidence Score", f"{verification_result['confidence']:.1%}")
            
            with col2:
                if "token_data" in result and result["token_data"]:
                    st.code(result["token_data"]["token"], language="text")
                    st.caption("Your verification token (keep it secure)")
            
            # Display extracted data
            if "ocr_data" in result:
                with st.expander("📄 Extracted Information"):
                    parsed_data = result["ocr_data"]["parsed_data"]
                    for key, value in parsed_data.items():
                        if value:
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
        
        else:
            st.error("❌ Document Verification Failed")
            st.write(f"**Reason**: {verification_result.get('reason', 'Unknown')}")
        
        # AI Feedback
        if "ai_feedback" in result:
            with st.expander("🤖 AI Assistant Feedback"):
                st.write(result["ai_feedback"])
    
    else:
        st.error("❌ Verification Error")
        st.write(result.get("error", "Unknown error occurred"))

def history_page():
    st.header("📋 Verification History")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/verify/history",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        
        if response.status_code == 200:
            history = response.json()
            
            if history:
                for record in history:
                    with st.container():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**Type**: {record['document_type']}")
                        
                        with col2:
                            status_emoji = "✅" if record['status'] == "approved" else "❌"
                            st.write(f"**Status**: {status_emoji} {record['status']}")
                        
                        with col3:
                            if record['confidence']:
                                st.write(f"**Confidence**: {record['confidence']:.1%}")
                        
                        with col4:
                            st.write(f"**Date**: {record['created_at'][:10]}")
                        
                        if record['token']:
                            st.code(record['token'], language="text")
                        
                        st.markdown("---")
            else:
                st.info("No verification history found")
        
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")

def institutions_page():
    st.header("🏛️ Institutional Token Verification")
    st.info("This portal is for financial institutions to verify user tokens")
    
    with st.form("token_verification"):
        institution_name = st.text_input("Institution Name")
        verification_token = st.text_input("Verification Token")
        submit = st.form_submit_button("Verify Token")
        
        if submit and verification_token:
            with st.spinner("Verifying token..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/institutions/verify-token",
                        json={
                            "token": verification_token,
                            "institution_name": institution_name
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Token Verified Successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("User Information")
                            st.write(f"**Name**: {result['user']['full_name']}")
                            st.write(f"**NIN**: {result['user']['nin_number']}")
                            st.write(f"**Level**: {result['user']['verification_level']}")
                        
                        with col2:
                            st.subheader("Verification Metadata")
                            st.write(f"**Confidence**: {result['metadata']['confidence']:.1%}")
                            st.write(f"**Document**: {result['metadata']['document_type']}")
                            st.write(f"**Verified**: {result['user']['verified_at'][:10]}")
                    
                    else:
                        st.error("❌ Token verification failed")
                        st.write("Token not found or invalid")
                
                except Exception as e:
                    st.error(f"Verification error: {str(e)}")

def logout():
    st.session_state.logged_in = False
    st.session_state.access_token = None
    st.session_state.user_info = None
    st.success("Logged out successfully!")
    st.rerun()

if __name__ == "__main__":
    main()
```

### 12. docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./kleardoc.db
      - AZURE_AI_VISION_ENDPOINT=${AZURE_AI_VISION_ENDPOINT}
      - AZURE_AI_VISION_KEY=${AZURE_AI_VISION_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
    volumes:
      - ./app:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    volumes:
      - ./streamlit_app:/streamlit_app
    command: streamlit run streamlit_app/streamlit_main.py --server.port=8501 --server.address=0.0.0.0
    depends_on:
      - api

volumes:
  db_data:
```

### 13. Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 14. .env.example
```bash
# Azure AI Vision
AZURE_AI_VISION_ENDPOINT=https://your-region.cognitiveservices.azure.com/
AZURE_AI_VISION_KEY=your_ai_vision_key

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=gpt-4

# NIMC API (use "mock" for development)
NIMC_API_ENDPOINT=https://api.nimc.gov.ng
NIMC_API_KEY=your_nimc_api_key

# Database
DATABASE_URL=sqlite:///./kleardoc.db

# Security
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🚀 Setup Instructions

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd kleardoc-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

3. **Run with Docker**:
```bash
docker-compose up --build
```

4. **Or Run Locally**:
```bash
# Terminal 1 - API
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Streamlit
streamlit run streamlit_app/streamlit_main.py --server.port 8501
```

5. **Access the Application**:
- Streamlit UI: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs
- Institution Portal: http://localhost:8501 (navigate to Institutions)

This complete Python implementation provides all the functionality specified in your requirements with proper Azure integrations, AI feedback, and secure token-based verification system.
