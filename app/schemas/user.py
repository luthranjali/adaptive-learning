"""User schemas for API serialization."""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    first_name: str
    last_name: str
    role: str = "student"
    grade_level: Optional[str] = None
    learning_goals: Optional[str] = None
    preferred_learning_style: Optional[str] = None


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for user updates."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    grade_level: Optional[str] = None
    learning_goals: Optional[str] = None
    preferred_learning_style: Optional[str] = None


class Token(BaseModel):
    """Schema for authentication token."""
    access_token: str
    token_type: str
