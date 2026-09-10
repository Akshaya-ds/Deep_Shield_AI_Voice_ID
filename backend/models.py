from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from datetime import datetime

from database import Base


# =========================================================
# ORGANIZATION
# =========================================================

class Organization(Base):

    __tablename__ = "organizations"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    organization = Column(
        String,
        nullable=False
    )

    domain = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    official_email = Column(
        String,
        nullable=False
    )

    domain_token = Column(
        String,
        nullable=False
    )

    email_verified = Column(
        Boolean,
        default=False
    )

    domain_verified = Column(
        Boolean,
        default=False
    )

    admin_approved = Column(
        Boolean,
        default=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================================
# AI AGENT
# =========================================================

class Agent(Base):

    __tablename__ = "agents"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    agent_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    organization = Column(
        String,
        nullable=False
    )

    agent_name = Column(
        String,
        nullable=False
    )

    # Cryptographic public identity
    public_key = Column(
        String,
        nullable=False
    )

    # =====================================================
    # REAL REGISTERED AI VOICE
    # =====================================================
    # Stores MFCC feature vector as JSON text.
    # This is used for voice comparison.

    voice_fingerprint = Column(
        Text,
        nullable=True
    )

    status = Column(
        String,
        default="PENDING"
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================================
# EMAIL OTP
# =========================================================

class EmailOTP(Base):

    __tablename__ = "email_otps"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    email = Column(
        String,
        index=True,
        nullable=False
    )

    otp_hash = Column(
        String,
        nullable=False
    )

    expires_at = Column(
        DateTime,
        nullable=False
    )

    attempts = Column(
        Integer,
        default=0
    )

    verified = Column(
        Boolean,
        default=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================================
# VERIFICATION SESSION
# =========================================================

class VerificationSession(Base):

    __tablename__ = "verification_sessions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    agent_id = Column(
        String,
        nullable=False
    )

    challenge = Column(
        String,
        nullable=False
    )

    expires_at = Column(
        DateTime,
        nullable=False
    )

    status = Column(
        String,
        default="PENDING"
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================================
# VERIFICATION LOG
# =========================================================

class VerificationLog(Base):

    __tablename__ = "verification_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        String,
        nullable=False
    )

    identity_status = Column(
        String,
        nullable=False
    )

    voice_score = Column(
        Float,
        nullable=True
    )

    risk_score = Column(
        Float,
        nullable=True
    )

    final_status = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )