"""User model for the adaptive learning assessment system."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    """User model for students and instructors."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default="student")  # student, instructor, admin
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Student-specific fields
    grade_level = Column(String(20), nullable=True)
    learning_goals = Column(Text, nullable=True)
    preferred_learning_style = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Additional profile data
    profile_data = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
