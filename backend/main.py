import os
import uuid
import smtplib
import dns.resolver
import re

from datetime import datetime, timedelta
from email.message import EmailMessage

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr

from sqlalchemy.orm import Session

from database import (
    Base,
    engine,
    get_db
)

from models import (
    Organization,
    Agent,
    EmailOTP
)

from crypto import (
    generate_identity,
    generate_otp,
    hash_otp,
    generate_domain_token
)

# ============================================================
# VOICE FUNCTIONS
# ============================================================

from voice import (
    generate_embedding,
    embedding_to_json,
    embedding_from_json,
    analyze_voice
)

from speech_to_text import (
    transcribe_audio
)

from organization_detector import (
    detect_organization
)

import numpy as np


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="DeepShield API",
    version="2.4.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# REQUEST MODELS
# ============================================================

class OrganizationRequest(BaseModel):

    organization: str
    domain: str
    official_email: EmailStr


class OTPRequest(BaseModel):

    email: EmailStr


class OTPVerifyRequest(BaseModel):

    email: EmailStr
    otp: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message": "DeepShield API is running",

        "status": "online",

        "version": "2.4.0"
    }


# ============================================================
# GET ALL AGENTS
# ============================================================

@app.get("/api/agents")
def get_agents(
    db: Session = Depends(get_db)
):

    agents = (
        db.query(Agent)
        .all()
    )

    return agents


# ============================================================
# GET SINGLE AGENT
# ============================================================

@app.get("/api/agents/{agent_id}")
def get_agent(
    agent_id: str,
    db: Session = Depends(get_db)
):

    agent = (
        db.query(Agent)
        .filter(
            Agent.agent_id == agent_id
        )
        .first()
    )

    if not agent:

        raise HTTPException(
            status_code=404,
            detail="Agent not found."
        )

    return agent


# ============================================================
# REGISTER ORGANIZATION
# ============================================================

@app.post("/api/organizations/register")
def register_organization(
    request: OrganizationRequest,
    db: Session = Depends(get_db)
):

    domain = (
        request.domain
        .lower()
        .strip()
    )

    domain = (
        domain
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .strip("/")
    )

    existing = (
        db.query(Organization)
        .filter(
            Organization.domain == domain
        )
        .first()
    )

    if existing:

        raise HTTPException(
            status_code=400,
            detail=(
                "This organization domain "
                "is already registered."
            )
        )

    domain_token = generate_domain_token()

    organization = Organization(

        organization=request.organization,

        domain=domain,

        official_email=str(
            request.official_email
        ).lower().strip(),

        domain_token=domain_token,

        email_verified=False,

        domain_verified=False,

        admin_approved=False
    )

    db.add(
        organization
    )

    db.commit()

    db.refresh(
        organization
    )

    return {

        "message": (
            "Organization registration started."
        ),

        "organization_id": organization.id,

        "organization": organization.organization,

        "domain": organization.domain,

        "official_email": organization.official_email,

        "verification_record": (
            f"deepshield-verification="
            f"{domain_token}"
        ),

        "email_verified": False,

        "domain_verified": False,

        "admin_approved": False,

        "status": "PENDING_VERIFICATION"
    }


# ============================================================
# SEND EMAIL OTP
# ============================================================

@app.post("/api/auth/send-otp")
def send_otp(
    request: OTPRequest,
    db: Session = Depends(get_db)
):

    email = (
        str(request.email)
        .lower()
        .strip()
    )

    otp = generate_otp()

    otp_hash = hash_otp(
        otp
    )

    expires_at = (
        datetime.utcnow()
        + timedelta(minutes=5)
    )

    db.query(
        EmailOTP
    ).filter(
        EmailOTP.email == email,
        EmailOTP.verified == False
    ).delete()

    record = EmailOTP(

        email=email,

        otp_hash=otp_hash,

        expires_at=expires_at,

        attempts=0,

        verified=False
    )

    db.add(
        record
    )

    db.commit()

    smtp_host = os.getenv(
        "SMTP_HOST"
    )

    smtp_port = int(
        os.getenv(
            "SMTP_PORT",
            "587"
        )
    )

    smtp_username = os.getenv(
        "SMTP_USERNAME"
    )

    smtp_password = os.getenv(
        "SMTP_PASSWORD"
    )

    if not all([
        smtp_host,
        smtp_username,
        smtp_password
    ]):

        db.delete(
            record
        )

        db.commit()

        raise HTTPException(
            status_code=500,
            detail="SMTP is not configured."
        )

    message = EmailMessage()

    message["Subject"] = (
        "DeepShield Email Verification"
    )

    message["From"] = smtp_username

    message["To"] = email

    message.set_content(
        f"""
DeepShield Email Verification

Your verification code is:

{otp}

This code expires in 5 minutes.

If you did not request this code,
please ignore this email.

DeepShield Security
"""
    )

    try:

        with smtplib.SMTP(
            smtp_host,
            smtp_port
        ) as server:

            server.starttls()

            server.login(
                smtp_username,
                smtp_password
            )

            server.send_message(
                message
            )

    except Exception as error:

        db.delete(
            record
        )

        db.commit()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Email delivery failed: {str(error)}"
            )
        )

    return {

        "message": (
            "Verification OTP sent."
        ),

        "expires_in_seconds": 300
    }


# ============================================================
# VERIFY EMAIL OTP
# ============================================================

@app.post("/api/auth/verify-otp")
def verify_otp(
    request: OTPVerifyRequest,
    db: Session = Depends(get_db)
):

    email = (
        str(request.email)
        .lower()
        .strip()
    )

    record = (
        db.query(EmailOTP)
        .filter(
            EmailOTP.email == email,
            EmailOTP.verified == False
        )
        .order_by(
            EmailOTP.created_at.desc()
        )
        .first()
    )

    if not record:

        raise HTTPException(
            status_code=400,
            detail=(
                "No active verification "
                "code found."
            )
        )

    if datetime.utcnow() > record.expires_at:

        raise HTTPException(
            status_code=400,
            detail="Verification code expired."
        )

    if record.attempts >= 5:

        raise HTTPException(
            status_code=429,
            detail=(
                "Too many incorrect attempts."
            )
        )

    record.attempts += 1

    if hash_otp(
        request.otp
    ) != record.otp_hash:

        db.commit()

        raise HTTPException(
            status_code=400,
            detail="Invalid verification code."
        )

    record.verified = True

    db.commit()

    organizations = (
        db.query(Organization)
        .filter(
            Organization.official_email == email
        )
        .all()
    )

    for organization in organizations:

        organization.email_verified = True

    db.commit()

    return {

        "message": (
            "Email verified successfully."
        ),

        "email_verified": True
    }


# ============================================================
# VERIFY DOMAIN OWNERSHIP
# ============================================================

@app.post(
    "/api/organizations/{organization_id}/verify-domain"
)
def verify_domain(
    organization_id: int,
    db: Session = Depends(get_db)
):

    organization = (
        db.query(Organization)
        .filter(
            Organization.id == organization_id
        )
        .first()
    )

    if not organization:

        raise HTTPException(
            status_code=404,
            detail="Organization not found."
        )

    expected_token = (
        f"deepshield-verification="
        f"{organization.domain_token}"
    )

    try:

        records = dns.resolver.resolve(
            organization.domain,
            "TXT"
        )

        found = False

        for record in records:

            value = str(
                record
            ).strip('"')

            if expected_token in value:

                found = True

                break

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not find the DeepShield "
                "verification TXT record."
            )
        )

    if not found:

        raise HTTPException(
            status_code=400,
            detail=(
                "Domain ownership could not be "
                "verified. Add the DeepShield TXT "
                "record first."
            )
        )

    organization.domain_verified = True

    db.commit()

    return {

        "message": (
            "Domain ownership verified."
        ),

        "domain": organization.domain,

        "domain_verified": True
    }


# ============================================================
# REGISTER AI AGENT + REAL REFERENCE VOICE
# ============================================================

@app.post("/api/agents/register")
async def register_agent(

    organization: str = Form(...),

    domain: str = Form(...),

    agent_name: str = Form(...),

    voice_file: UploadFile = File(...),

    db: Session = Depends(get_db)
):

    # ========================================================
    # 1. CLEAN DOMAIN
    # ========================================================

    domain = (
        domain
        .lower()
        .strip()
    )

    domain = (
        domain
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .strip("/")
    )

    # ========================================================
    # 2. FIND ORGANIZATION
    # ========================================================

    organization_record = (
        db.query(Organization)
        .filter(
            Organization.domain == domain
        )
        .first()
    )

    if not organization_record:

        raise HTTPException(
            status_code=404,
            detail="Organization not found."
        )

    # ========================================================
    # 3. SECURITY CHECK
    # ========================================================

    if not organization_record.email_verified:

        raise HTTPException(
            status_code=403,
            detail=(
                "Organization email "
                "is not verified."
            )
        )

    # ========================================================
    # 4. VALIDATE VOICE
    # ========================================================

    allowed_types = [

        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mp4",
        "audio/x-m4a"
    ]

    if voice_file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid audio format. "
                "Upload MP3, WAV or M4A."
            )
        )

    audio_data = await voice_file.read()

    max_size = (
        20 * 1024 * 1024
    )

    if len(audio_data) > max_size:

        raise HTTPException(
            status_code=400,
            detail=(
                "Audio file must be "
                "smaller than 20 MB."
            )
        )

    if len(audio_data) == 0:

        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    # ========================================================
    # 5. GENERATE IDENTITY
    # ========================================================

    identity = generate_identity()

    # ========================================================
    # 6. AGENT ID
    # ========================================================

    agent_number = (
        db.query(Agent).count()
        + 1
    )

    agent_id = (
        f"DS-{agent_number:06d}"
    )

    # ========================================================
    # 7. STORAGE
    # ========================================================

    storage_directory = os.path.join(
        "storage",
        "voices"
    )

    os.makedirs(
        storage_directory,
        exist_ok=True
    )

    extension = ".mp3"

    if voice_file.filename:

        original_extension = (
            os.path.splitext(
                voice_file.filename
            )[1]
            .lower()
        )

        if original_extension in [
            ".mp3",
            ".wav",
            ".m4a"
        ]:

            extension = (
                original_extension
            )

    temp_filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    temp_path = os.path.join(
        storage_directory,
        temp_filename
    )

    # ========================================================
    # 8. CREATE REFERENCE VOICE
    # ========================================================

    try:

        with open(
            temp_path,
            "wb"
        ) as buffer:

            buffer.write(
                audio_data
            )

        embedding = generate_embedding(
            temp_path
        )

        voice_fingerprint = (
            embedding_to_json(
                embedding
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Voice processing failed: "
                f"{str(error)}"
            )
        )

    finally:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )

    # ========================================================
    # 9. CREATE AGENT
    # ========================================================

    agent = Agent(

        agent_id=agent_id,

        organization=(
            organization_record.organization
        ),

        agent_name=agent_name,

        public_key=(
            identity["public_key"]
        ),

        voice_fingerprint=(
            voice_fingerprint
        ),

        status="VERIFIED"
    )

    db.add(
        agent
    )

    db.commit()

    db.refresh(
        agent
    )

    return {

        "message": (
            "AI Agent and reference voice "
            "registered successfully."
        ),

        "agent_id": agent.agent_id,

        "organization": agent.organization,

        "agent_name": agent.agent_name,

        "voice_registered": True,

        "voice_embedding_created": True,

        "public_key": agent.public_key,

        "status": agent.status
    }


# ============================================================
# ============================================================
# CONVERSATION RISK ANALYSIS
# ============================================================
# ============================================================
#
# IMPORTANT:
#
# This is NOT a simple:
#
#     if "KYC" in transcript:
#         scam
#
# Instead, DeepShield looks for combinations of:
#
#   - urgency
#   - threats
#   - financial consequences
#   - credential requests
#   - OTP/PIN/password requests
#   - money transfer requests
#   - account blocking
#   - suspicious verification requests
#   - impersonation
#
# The more dangerous signals occur together,
# the higher the conversation risk.
#
# This is an explainable MVP risk engine.
# ============================================================


def _contains_any(
    text: str,
    patterns
):
    """
    Return True if any pattern is found.
    """

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            return True

    return False


def analyze_conversation_risk(
    transcript: str
):
    """
    Analyze transcript for scam / manipulation risk.

    Returns:

        {
            risk_score,
            risk_level,
            risk_flags,
            detected_signals
        }

    Risk score:
        0   -> very low risk
        1-29 -> low
        30-59 -> medium
        60-79 -> high
        80-100 -> critical
    """

    if not transcript:

        return {

            "risk_score": 0,

            "risk_level": "LOW",

            "risk_flags": [],

            "detected_signals": [],

            "message": (
                "No transcript available "
                "for conversation risk analysis."
            )
        }

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    text = (
        transcript
        .lower()
        .strip()
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    risk_score = 0

    flags = []

    detected_signals = []

    # ========================================================
    # SIGNAL GROUPS
    # ========================================================

    urgency_patterns = [

        r"\bimmediately\b",

        r"\burgent(?:ly)?\b",

        r"\bright away\b",

        r"\bact now\b",

        r"\bdo it now\b",

        r"\bwithin (?:\d+|one|two|three|five|ten) minutes?\b",

        r"\bwithout delay\b",

        r"\bas soon as possible\b",

        r"\bnow\b",

        r"\blast warning\b",

        r"\bfinal warning\b"
    ]

    threat_patterns = [

        r"\baccount will be blocked\b",

        r"\baccount will be closed\b",

        r"\baccount will be suspended\b",

        r"\baccount will be frozen\b",

        r"\bcard will be blocked\b",

        r"\bservices will be stopped\b",

        r"\blegal action\b",

        r"\bpolice complaint\b",

        r"\barrest\b",

        r"\byou will lose\b",

        r"\byour account.*(?:blocked|closed|frozen|suspended)\b",

        r"\botherwise\b",

        r"\bif you don't\b",

        r"\bif you do not\b"
    ]

    financial_patterns = [

        r"(?:₹|rs\.?|inr)\s*\d[\d,]*",

        r"\b\d+\s*(?:lakh|lakhs|crore|crores)\b",

        r"\bone lakh\b",

        r"\btwo lakh\b",

        r"\bfinancial loss\b",

        r"\bmoney will be deducted\b",

        r"\bamount will be deducted\b",

        r"\bamount.*deduct\b",

        r"\bpenalty\b",

        r"\bfine\b",

        r"\bpay.*immediately\b",

        r"\bpayment.*immediately\b"
    ]

    credential_patterns = [

        r"\botp\b",

        r"\bone time password\b",

        r"\bpin\b",

        r"\bpassword\b",

        r"\bpasscode\b",

        r"\bcvv\b",

        r"\bcard number\b",

        r"\bdebit card details\b",

        r"\bcredit card details\b",

        r"\bnet banking password\b",

        r"\bsecurity code\b"
    ]

    sensitive_request_patterns = [

        r"\bshare\b.*\botp\b",

        r"\bsend\b.*\botp\b",

        r"\btell me\b.*\botp\b",

        r"\bgive me\b.*\botp\b",

        r"\bprovide\b.*\botp\b",

        r"\bshare\b.*\bpin\b",

        r"\bshare\b.*\bpassword\b",

        r"\btell me\b.*\bpassword\b",

        r"\bprovide\b.*\bpassword\b",

        r"\bshare\b.*\bcard\b.*\bdetails\b",

        r"\bsend\b.*\bcard\b.*\bdetails\b"
    ]

    money_transfer_patterns = [

        r"\btransfer\b.*\bmoney\b",

        r"\btransfer\b.*\bfunds\b",

        r"\bsend\b.*\bmoney\b",

        r"\bsend\b.*\bfunds\b",

        r"\bmake\b.*\bpayment\b",

        r"\bpay\b.*\baccount\b",

        r"\bdeposit\b.*\baccount\b",

        r"\bupi\b",

        r"\bupi id\b",

        r"\bbeneficiary\b",

        r"\bbank transfer\b",

        r"\btransfer\b.*(?:₹|rs\.?|inr|\d+\s*(?:lakh|lakhs|crore|crores)?|one\s+lakh|two\s+lakh|three\s+lakh)",

        r"\bsend\b.*(?:₹|rs\.?|inr|\d+\s*(?:lakh|lakhs|crore|crores)?|one\s+lakh|two\s+lakh|three\s+lakh)"
    ]

    kyc_patterns = [

        r"\bkyc\b",

        r"\bknow your customer\b",

        r"\bverification\b",

        r"\bverify your account\b",

        r"\bverify account\b",

        r"\bidentity verification\b"
    ]

    # KYC itself is not automatically a scam.
    # We separately detect whether the caller is actually
    # demanding an immediate KYC action from the customer.
    kyc_action_patterns = [

        r"\bcomplete (?:your )?kyc\b",

        r"\bupdate (?:your )?kyc\b",

        r"\bdo (?:your )?kyc\b",

        r"\bfinish (?:your )?kyc\b",

        r"\bkyc (?:is )?(?:pending|required|due)\b",

        r"\bkyc.*(?:complete|update|verify|submit)\b",

        r"\bverify.*(?:account|identity).*now\b",

        r"\bcomplete.*verification.*now\b",

        r"\bupdate.*details.*for.*kyc\b"
    ]

    impersonation_patterns = [

        r"\bwe are from\b",

        r"\bcalling from\b",

        r"\bthis is from\b",

        r"\bfrom your bank\b",

        r"\bfrom the bank\b",

        r"\bbank official\b",

        r"\bcustomer care\b",

        r"\bsecurity department\b",

        r"\bcyber security team\b"
    ]

    remote_access_patterns = [

        r"\banydesk\b",

        r"\bteamviewer\b",

        r"\bremote access\b",

        r"\bscreen sharing\b",

        r"\bshare your screen\b",

        r"\binstall this app\b",

        r"\bdownload this app\b"
    ]

    suspicious_link_patterns = [

        r"\bclick this link\b",

        r"\bclick the link\b",

        r"\bopen this link\b",

        r"\bverify using this link\b",

        r"\buse this link\b",

        r"\blink sent\b"
    ]

    # ========================================================
    # DETECT INDIVIDUAL SIGNALS
    # ========================================================

    has_urgency = _contains_any(
        text,
        urgency_patterns
    )

    has_threat = _contains_any(
        text,
        threat_patterns
    )

    has_financial = _contains_any(
        text,
        financial_patterns
    )

    has_credentials = _contains_any(
        text,
        credential_patterns
    )

    has_sensitive_request = _contains_any(
        text,
        sensitive_request_patterns
    )

    has_transfer = _contains_any(
        text,
        money_transfer_patterns
    )

    has_kyc = _contains_any(
        text,
        kyc_patterns
    )

    has_kyc_action = _contains_any(
        text,
        kyc_action_patterns
    )

    has_impersonation = _contains_any(
        text,
        impersonation_patterns
    )

    has_remote_access = _contains_any(
        text,
        remote_access_patterns
    )

    has_link = _contains_any(
        text,
        suspicious_link_patterns
    )

    # ========================================================
    # BASE SCORES
    # ========================================================

    if has_urgency:

        risk_score += 12

        flags.append(
            "Urgency / pressure detected"
        )

        detected_signals.append(
            "URGENCY"
        )

    if has_threat:

        risk_score += 22

        flags.append(
            "Threat or negative consequence detected"
        )

        detected_signals.append(
            "THREAT"
        )

    if has_financial:

        risk_score += 20

        flags.append(
            "Financial loss / money-related language detected"
        )

        detected_signals.append(
            "FINANCIAL_RISK"
        )

    if has_credentials:

        risk_score += 18

        flags.append(
            "Sensitive credential language detected"
        )

        detected_signals.append(
            "CREDENTIAL_REQUEST"
        )

    if has_sensitive_request:

        risk_score += 22

        flags.append(
            "Request for sensitive authentication information detected"
        )

        detected_signals.append(
            "SENSITIVE_DATA_REQUEST"
        )

    if has_transfer:

        risk_score += 22

        flags.append(
            "Money transfer / payment instruction detected"
        )

        detected_signals.append(
            "MONEY_TRANSFER"
        )

    if has_kyc:

        # Mentioning KYC in a normal support conversation
        # is not enough to label the call as a scam.
        risk_score += 2

        detected_signals.append(
            "KYC_OR_VERIFICATION"
        )

    if has_kyc_action:

        # An actual KYC action request is materially different
        # from simply discussing KYC. For this MVP, an unsolicited
        # request to complete/update/verify KYC is treated as a
        # strong risk signal because it asks the customer to take
        # an account-security action during a voice interaction.
        risk_score += 60

        flags.append(
            "KYC action / verification request detected"
        )

        detected_signals.append(
            "KYC_ACTION_REQUEST"
        )

    if has_impersonation:

        risk_score += 5

        detected_signals.append(
            "IMPERSONATION_LANGUAGE"
        )

    if has_remote_access:

        risk_score += 25

        flags.append(
            "Remote-access / screen-sharing request detected"
        )

        detected_signals.append(
            "REMOTE_ACCESS"
        )

    if has_link:

        risk_score += 12

        flags.append(
            "Suspicious link/action instruction detected"
        )

        detected_signals.append(
            "SUSPICIOUS_LINK"
        )

    # ========================================================
    # CONTEXTUAL COMBINATIONS
    # ========================================================
    #
    # This is the most important part.
    #
    # Example:
    #
    # KYC alone
    #       -> low
    #
    # KYC + urgency
    #       -> higher
    #
    # KYC + urgency + financial threat
    #       -> critical
    #
    # ========================================================

    if has_kyc_action and has_urgency:

        risk_score += 25

        flags.append(
            "KYC request combined with urgent pressure"
        )

        detected_signals.append(
            "KYC_PRESSURE_COMBINATION"
        )

    if has_kyc_action and has_threat:

        risk_score += 25

        flags.append(
            "KYC request combined with account / service threat"
        )

        detected_signals.append(
            "KYC_THREAT_COMBINATION"
        )

    if (
        has_kyc_action
        and has_urgency
        and has_financial
    ):

        risk_score += 30

        flags.append(
            "KYC + urgency + financial-loss threat"
        )

        detected_signals.append(
            "HIGH_RISK_KYC_SCAM_PATTERN"
        )

    if (
        has_credentials
        and has_sensitive_request
    ):

        risk_score += 25

        flags.append(
            "Credential request combined with direct information request"
        )

        detected_signals.append(
            "CREDENTIAL_HARVESTING_PATTERN"
        )

    if (
        has_urgency
        and has_financial
        and has_threat
    ):

        risk_score += 25

        flags.append(
            "Urgency + financial loss + threat combination"
        )

        detected_signals.append(
            "FINANCIAL_THREAT_PATTERN"
        )

    if (
        has_transfer
        and has_urgency
    ):

        risk_score += 25

        flags.append(
            "Urgent money-transfer instruction"
        )

        detected_signals.append(
            "URGENT_TRANSFER_PATTERN"
        )

    if (
        has_credentials
        and has_sensitive_request
        and has_urgency
    ):

        risk_score += 25

        flags.append(
            "Urgent request for authentication credentials"
        )

        detected_signals.append(
            "URGENT_CREDENTIAL_REQUEST"
        )

    if (
        has_transfer
        and has_financial
        and has_urgency
    ):

        risk_score += 20

        flags.append(
            "Urgent transfer involving money or financial loss"
        )

        detected_signals.append(
            "URGENT_FINANCIAL_TRANSFER_PATTERN"
        )

    if (
        has_impersonation
        and has_urgency
        and (
            has_credentials
            or has_transfer
            or has_financial
        )
    ):

        risk_score += 20

        flags.append(
            "Possible impersonation combined with a high-risk request"
        )

        detected_signals.append(
            "IMPERSONATION_SCAM_PATTERN"
        )

    if (
        has_remote_access
        and has_urgency
    ):

        risk_score += 20

        flags.append(
            "Urgent remote-access request"
        )

        detected_signals.append(
            "REMOTE_ACCESS_SCAM_PATTERN"
        )

    # ========================================================
    # IMPORTANT BENIGN CONTEXT
    # ========================================================
    #
    # These phrases indicate normal support-like
    # conversation.
    #
    # They do NOT automatically cancel risk.
    #
    # They only prevent harmless conversations
    # from receiving unnecessary risk.
    # ========================================================

    benign_patterns = [

        r"\bhow can i help\b",

        r"\bhow may i help\b",

        r"\bi can help you\b",

        r"\bi can assist you\b",

        r"\bgeneral banking information\b",

        r"\baccount services\b",

        r"\btransaction quer(?:y|ies)\b",

        r"\bplease let me know\b",

        r"\bhow can we assist\b",

        r"\bthank you for contacting\b"
    ]

    benign_context = _contains_any(
        text,
        benign_patterns
    )

    if benign_context:

        detected_signals.append(
            "BENIGN_SUPPORT_CONTEXT"
        )

        # Only apply a small reduction.
        # Never cancel a strong scam signal.
        if risk_score < 20:

            risk_score = max(
                0,
                risk_score - 5
            )

    # ========================================================
    # CAP SCORE
    # ========================================================

    risk_score = int(
        max(
            0,
            min(
                100,
                risk_score
            )
        )
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk_score >= 80:

        risk_level = "CRITICAL"

    elif risk_score >= 60:

        risk_level = "HIGH"

    elif risk_score >= 30:

        risk_level = "MEDIUM"

    elif risk_score >= 1:

        risk_level = "LOW"

    else:

        risk_level = "VERY_LOW"

    # ========================================================
    # MESSAGE
    # ========================================================

    if risk_level == "CRITICAL":

        message = (
            "Multiple high-risk scam indicators were "
            "detected in the conversation."
        )

    elif risk_level == "HIGH":

        message = (
            "The conversation contains significant "
            "fraud or manipulation indicators."
        )

    elif risk_level == "MEDIUM":

        message = (
            "The conversation contains some suspicious "
            "signals and should be reviewed."
        )

    elif risk_level == "LOW":

        message = (
            "A small number of potentially sensitive "
            "signals were detected."
        )

    else:

        message = (
            "No significant scam-risk pattern was detected."
        )

    return {

        "risk_score": risk_score,

        "risk_level": risk_level,

        "risk_flags": flags,

        "detected_signals": list(
            dict.fromkeys(
                detected_signals
            )
        ),

        "message": message
    }


# ============================================================
# ACOUSTIC SCORE
# ============================================================

def calculate_acoustic_score(
    voice_analysis
):
    """
    Convert the acoustic analysis into one explainable
    acoustic score.

    This does NOT replace WavLM.

    It is only a supporting signal.
    """

    if not voice_analysis:

        return 50.0

    pitch = voice_analysis.get(
        "pitch",
        {}
    )

    energy = voice_analysis.get(
        "energy",
        {}
    )

    pauses = voice_analysis.get(
        "pauses",
        {}
    )

    speech_consistency = float(
        voice_analysis.get(
            "speech_consistency",
            50.0
        )
    )

    naturalness = float(
        voice_analysis.get(
            "naturalness_score",
            50.0
        )
    )

    # ========================================================
    # PITCH SCORE
    # ========================================================

    pitch_variation = float(
        pitch.get(
            "pitch_variation",
            0.0
        )
    )

    if 0.03 <= pitch_variation <= 0.35:

        pitch_score = 100.0

    elif pitch_variation < 0.03:

        pitch_score = (
            pitch_variation
            /
            0.03
            *
            100
        )

    else:

        pitch_score = max(
            0.0,
            100.0
            -
            (
                pitch_variation
                -
                0.35
            )
            * 180
        )

    # ========================================================
    # ENERGY SCORE
    # ========================================================

    energy_score = float(
        energy.get(
            "energy_consistency",
            50.0
        )
    )

    # ========================================================
    # PAUSE SCORE
    # ========================================================

    silence_ratio = float(
        pauses.get(
            "silence_ratio",
            0.0
        )
    )

    if 2 <= silence_ratio <= 45:

        pause_score = 100.0

    elif silence_ratio < 2:

        pause_score = (
            silence_ratio
            /
            2
            *
            100
        )

    else:

        pause_score = max(
            0.0,
            100.0
            -
            (
                silence_ratio
                -
                45
            )
            * 1.8
        )

    # ========================================================
    # FINAL ACOUSTIC SCORE
    # ========================================================

    acoustic_score = (

        np.clip(
            pitch_score,
            0,
            100
        )
        * 0.25

        +

        np.clip(
            energy_score,
            0,
            100
        )
        * 0.25

        +

        np.clip(
            pause_score,
            0,
            100
        )
        * 0.15

        +

        np.clip(
            speech_consistency,
            0,
            100
        )
        * 0.35
    )

    # Small contribution from naturalness,
    # because naturalness is already derived
    # from acoustic signals.

    acoustic_score = (
        acoustic_score * 0.75
        +
        naturalness * 0.25
    )

    return round(
        float(
            np.clip(
                acoustic_score,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# FINAL TRUST SCORE
# ============================================================

def calculate_final_trust_score(
    speaker_similarity,
    naturalness_score,
    acoustic_score,
    conversation_risk,
    organization_match
):
    """
    Calculate DeepShield's final trust score.

    Design principle:
        IDENTITY and CONVERSATION SAFETY are separate signals.

    A strong registered voice match can produce high trust only
    when the conversation is also safe. A dangerous conversation
    can override an otherwise genuine voice match.

    Naturalness/acoustic metrics remain diagnostic only; they do
    not reduce a genuine registered voice merely because the
    recording quality or delivery style changed.
    """

    speaker_similarity = float(np.clip(speaker_similarity, 0, 100))
    naturalness_score = float(np.clip(naturalness_score, 0, 100))
    acoustic_score = float(np.clip(acoustic_score, 0, 100))
    conversation_risk = float(np.clip(conversation_risk, 0, 100))

    organization_score = 100.0 if organization_match else 0.0

    supporting_audio_score = (
        naturalness_score * 0.50
        + acoustic_score * 0.50
    )

    # ------------------------------------------------------------
    # 1. IDENTITY SCORE
    # ------------------------------------------------------------
    # Organization-first verification means that this score is
    # meaningful only after a registered organization/agent has
    # been found.
    #
    # Strong WavLM matches:
    #   95+ -> very strong identity evidence
    #   90-94 -> strong
    #   86-89 -> usable/review
    #   <86 -> not trusted as the registered voice
    # ------------------------------------------------------------

    if organization_match:
        identity_score = speaker_similarity
    else:
        identity_score = speaker_similarity * 0.50

    # ------------------------------------------------------------
    # 2. CONVERSATION SAFETY ADJUSTMENT
    # ------------------------------------------------------------
    # Normal conversation must preserve a high score.
    # Moderate concern moves a genuine identity into the
    # 70-80-ish review zone. Strong scam patterns can override
    # even a 99-100% voice match.
    # ------------------------------------------------------------

    if conversation_risk <= 10:
        # Clean/normal support conversation.
        # Preserve the genuine identity score.
        final_score = identity_score

    elif conversation_risk <= 25:
        # Mildly unusual or sensitive context.
        # Keep a strong identity match in the moderate/high range.
        progress = (conversation_risk - 10.0) / 15.0
        reduction = 8.0 * (progress ** 1.10)
        final_score = identity_score - reduction

    elif conversation_risk <= 45:
        # Noticeable suspicious context.
        # This is where a genuine voice should normally land around
        # 65-80 depending on the actual voice match.
        progress = (conversation_risk - 25.0) / 20.0
        target_factor = 0.82 - (0.12 * progress)
        final_score = identity_score * target_factor

    elif conversation_risk <= 65:
        # High-risk conversation. Voice identity is still visible,
        # but trust must fall substantially.
        progress = (conversation_risk - 45.0) / 20.0
        target_factor = 0.65 - (0.40 * progress)
        final_score = identity_score * target_factor

    else:
        # Critical scam-like conversation.
        # A cloned/genuine voice must NOT rescue the transaction.
        progress = (conversation_risk - 65.0) / 35.0
        target_factor = 0.25 - (0.20 * progress)
        final_score = identity_score * target_factor

    # ------------------------------------------------------------
    # 3. MISMATCH PROTECTION
    # ------------------------------------------------------------
    # WavLM can give surprisingly high similarities to unrelated
    # voices in some recordings. Do not let those become trusted.
    # ------------------------------------------------------------

    if speaker_similarity < 60:
        mismatch_penalty = speaker_similarity / 60.0
    elif speaker_similarity < 75:
        mismatch_penalty = 0.90
    else:
        mismatch_penalty = 1.0

    final_score *= mismatch_penalty

    # A non-verified voice must never look like a trusted registered
    # agent, even when the conversation is harmless.
    if speaker_similarity < 86:
        final_score = min(final_score, 55)

    final_score = int(round(float(np.clip(final_score, 0, 99))))

    return {
        "base_trust": round(identity_score, 2),
        "risk_multiplier": round(
            final_score / max(identity_score, 1.0),
            4
        ),
        "mismatch_penalty": round(mismatch_penalty, 4),
        "supporting_audio_score": round(supporting_audio_score, 2),
        "organization_score": organization_score,
        "final_trust_score": final_score
    }


# ============================================================
# TRUST LEVEL
# ============================================================

def get_trust_level(trust_score, conversation_risk, speaker_similarity):
    """Convert the final score and risk signals into a DeepShield level."""

    if conversation_risk >= 70 or trust_score <= 35:
        return "HIGH_RISK"

    if conversation_risk >= 50 or trust_score <= 55 or speaker_similarity < 75:
        return "SUSPICIOUS"

    if conversation_risk >= 25 or speaker_similarity < 86 or trust_score < 86:
        return "REVIEW"

    if trust_score >= 90 and conversation_risk <= 10:
        return "HIGH_TRUST"

    return "MODERATE_TRUST"


# ============================================================
# VERIFY UPLOADED VOICE
# ============================================================

@app.post("/api/verify-voice")
async def verify_voice(

    voice_file: UploadFile = File(...),

    db: Session = Depends(get_db)
):

    # ========================================================
    # 1. VALIDATE AUDIO
    # ========================================================

    allowed_types = [

        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mp4",
        "audio/x-m4a"
    ]

    if voice_file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid audio format. "
                "Upload MP3, WAV or M4A."
            )
        )

    audio_data = await voice_file.read()

    if not audio_data:

        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    if len(audio_data) > (
        20 * 1024 * 1024
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Audio file must be "
                "smaller than 20 MB."
            )
        )

    # ========================================================
    # 2. CREATE TEMP FILE
    # ========================================================

    storage_directory = os.path.join(
        "storage",
        "voices"
    )

    os.makedirs(
        storage_directory,
        exist_ok=True
    )

    extension = ".mp3"

    if voice_file.filename:

        original_extension = (
            os.path.splitext(
                voice_file.filename
            )[1]
            .lower()
        )

        if original_extension in [
            ".mp3",
            ".wav",
            ".m4a"
        ]:

            extension = original_extension

    temp_path = os.path.join(

        storage_directory,

        f"verify_{uuid.uuid4().hex}"
        f"{extension}"
    )

    try:

        with open(
            temp_path,
            "wb"
        ) as buffer:

            buffer.write(
                audio_data
            )

        # ====================================================
        # 3. TRANSCRIBE
        # ====================================================

        transcript = transcribe_audio(
            temp_path
        )

        # ====================================================
        # 4. ORGANIZATION DETECTION
        # ====================================================

        organizations = (
            db.query(Organization)
            .all()
        )

        organization_result = (
            detect_organization(
                transcript,
                organizations
            )
        )

        # ====================================================
        # CONVERSATION RISK
        # ====================================================

        risk_analysis = (
            analyze_conversation_risk(
                transcript
            )
        )

        conversation_risk = float(
            risk_analysis.get(
                "risk_score",
                0
            )
        )

        # ====================================================
        # NO ORGANIZATION
        # ====================================================

        if not organization_result:

            # ------------------------------------------------
            # Even without organization detection,
            # transcript risk is still useful.
            # ------------------------------------------------

            base_score = (
                40.0
            )

            risk_normalized = (
                conversation_risk
                /
                100.0
            )

            risk_multiplier = (
                1.0
                -
                (
                    0.80
                    *
                    (
                        risk_normalized
                        **
                        1.15
                    )
                )
            )

            risk_multiplier = max(
                0.20,
                min(
                    1.0,
                    risk_multiplier
                )
            )

            final_score = round(
                base_score
                *
                risk_multiplier
            )

            if conversation_risk >= 80:

                verdict = (
                    "HIGH_RISK_SCAM"
                )

            elif conversation_risk >= 60:

                verdict = (
                    "SUSPICIOUS_CONVERSATION"
                )

            else:

                verdict = (
                    "ORGANIZATION_NOT_DETECTED"
                )

            return {

                "success": True,

                "verdict": verdict,

                "trust_score": final_score,

                "similarity": 0,

                "raw_similarity": 0,

                "transcript": transcript,

                "organization_detected": False,

                "naturalness_score": 0,

                "voice_analysis": None,

                "risk_analysis": risk_analysis,

                "score_breakdown": {

                    "speaker_similarity": 0,

                    "naturalness": 0,

                    "acoustic_score": 0,

                    "conversation_risk":
                        conversation_risk,

                    "organization_match": 0,

                    "final_trust_score":
                        final_score
                },

                "message": (
                    "No registered organization "
                    "was detected in the uploaded voice."
                )
            }

        # ====================================================
        # 5. ORGANIZATION DETAILS
        # ====================================================

        detected_organization_id = (
            organization_result[
                "organization_id"
            ]
        )

        detected_organization = (
            organization_result[
                "organization"
            ]
        )

        # ====================================================
        # 6. FIND AGENTS
        # ====================================================

        agents = (
            db.query(Agent)
            .filter(
                Agent.status == "VERIFIED",
                Agent.organization ==
                detected_organization
            )
            .all()
        )

        # ====================================================
        # NO AGENT
        # ====================================================

        if not agents:

            # Organization is known,
            # but there is no reference voice.

            organization_score = 100.0

            base_score = (
                organization_score
                * 0.15
            )

            risk_multiplier = (
                1.0
                -
                (
                    0.80
                    *
                    (
                        (
                            conversation_risk
                            /
                            100
                        )
                        **
                        1.15
                    )
                )
            )

            risk_multiplier = max(
                0.20,
                min(
                    1.0,
                    risk_multiplier
                )
            )

            final_score = round(
                base_score
                *
                risk_multiplier
            )

            if conversation_risk >= 80:

                verdict = (
                    "HIGH_RISK_SCAM"
                )

            elif conversation_risk >= 60:

                verdict = (
                    "SUSPICIOUS_CONVERSATION"
                )

            else:

                verdict = (
                    "NO_REGISTERED_AGENT"
                )

            return {

                "success": True,

                "verdict": verdict,

                "trust_score": final_score,

                "similarity": 0,

                "raw_similarity": 0,

                "transcript": transcript,

                "organization_detected": True,

                "organization":
                    detected_organization,

                "naturalness_score": 0,

                "voice_analysis": None,

                "risk_analysis":
                    risk_analysis,

                "score_breakdown": {

                    "speaker_similarity": 0,

                    "naturalness": 0,

                    "acoustic_score": 0,

                    "conversation_risk":
                        conversation_risk,

                    "organization_match": 100,

                    "final_trust_score":
                        final_score
                },

                "message": (
                    f"{detected_organization} was detected, "
                    "but no verified AI agent is registered "
                    "for this organization."
                )
            }

        # ====================================================
        # 7. VOICE QUALITY / NATURALNESS ANALYSIS
        # ====================================================

        try:

            voice_analysis = analyze_voice(
                temp_path
            )

        except Exception as error:

            voice_analysis = {

                "naturalness_score": 50.0,

                "pitch": {

                    "pitch_variation": 0.0,

                    "voiced_ratio": 0.0,

                    "median_pitch": 0.0,

                    "pitch_range": 0.0
                },

                "energy": {

                    "energy_variation": 0.0,

                    "energy_dynamic_range": 0.0,

                    "energy_consistency": 50.0
                },

                "pauses": {

                    "silence_ratio": 0.0,

                    "pause_count": 0,

                    "average_pause_duration": 0.0,

                    "longest_pause": 0.0
                },

                "spectral": {

                    "spectral_centroid": 0.0,

                    "spectral_bandwidth": 0.0,

                    "spectral_rolloff": 0.0,

                    "zero_crossing_rate": 0.0,

                    "spectral_flatness": 0.0
                },

                "speech_consistency": 50.0,

                "analysis_error": str(error)
            }

        # ====================================================
        # 8. EXTRACT NATURALNESS
        # ====================================================

        naturalness_score = float(
            voice_analysis.get(
                "naturalness_score",
                50.0
            )
        )

        naturalness_score = max(
            0.0,
            min(
                100.0,
                naturalness_score
            )
        )

        # ====================================================
        # 9. ACOUSTIC SCORE
        # ====================================================

        acoustic_score = (
            calculate_acoustic_score(
                voice_analysis
            )
        )

        # ====================================================
        # 10. GENERATE UPLOADED EMBEDDING
        # ====================================================

        uploaded_embedding = (
            generate_embedding(
                temp_path
            )
        )

        uploaded_vector = np.asarray(
            uploaded_embedding,
            dtype=np.float32
        )

        # ====================================================
        # 11. COMPARE SPEAKERS
        # ====================================================

        results = []

        for agent in agents:

            if not agent.voice_fingerprint:

                continue

            try:

                stored_embedding = (
                    embedding_from_json(
                        agent.voice_fingerprint
                    )
                )

                stored_vector = np.asarray(
                    stored_embedding,
                    dtype=np.float32
                )

                if (
                    stored_vector.shape
                    != uploaded_vector.shape
                ):

                    continue

                stored_norm = np.linalg.norm(
                    stored_vector
                )

                uploaded_norm = np.linalg.norm(
                    uploaded_vector
                )

                if (
                    stored_norm == 0
                    or uploaded_norm == 0
                ):

                    continue

                cosine_similarity = float(

                    np.dot(
                        stored_vector,
                        uploaded_vector
                    )
                    /
                    (
                        stored_norm
                        *
                        uploaded_norm
                    )
                )

                raw_similarity = (

                    max(
                        0.0,
                        min(
                            1.0,
                            cosine_similarity
                        )
                    )
                    * 100
                )

                raw_similarity = round(
                    raw_similarity,
                    2
                )

                results.append({

                    "agent_id":
                        agent.agent_id,

                    "organization":
                        agent.organization,

                    "agent_name":
                        agent.agent_name,

                    "similarity":
                        raw_similarity
                })

            except Exception:

                continue

        # ====================================================
        # 12. NO COMPATIBLE EMBEDDING
        # ====================================================

        if not results:

            if conversation_risk >= 80:

                verdict = (
                    "HIGH_RISK_SCAM"
                )

            elif conversation_risk >= 60:

                verdict = (
                    "SUSPICIOUS_CONVERSATION"
                )

            else:

                verdict = (
                    "VOICE_COMPARISON_FAILED"
                )

            return {

                "success": True,

                "verdict": verdict,

                "trust_score": 0,

                "similarity": 0,

                "raw_similarity": 0,

                "transcript": transcript,

                "organization_detected": True,

                "organization":
                    detected_organization,

                "naturalness_score":
                    round(
                        naturalness_score,
                        2
                    ),

                "voice_analysis":
                    voice_analysis,

                "risk_analysis":
                    risk_analysis,

                "score_breakdown": {

                    "speaker_similarity": 0,

                    "naturalness":
                        round(
                            naturalness_score,
                            2
                        ),

                    "acoustic_score":
                        acoustic_score,

                    "conversation_risk":
                        conversation_risk,

                    "organization_match": 100,

                    "final_trust_score": 0
                },

                "message": (
                    "Organization was detected, "
                    "but the registered reference "
                    "voice could not be compared."
                )
            }

        # ====================================================
        # 13. BEST MATCH
        # ====================================================

        results.sort(
            key=lambda item:
                item["similarity"],
            reverse=True
        )

        best_match = results[0]

        raw_similarity = (
            best_match["similarity"]
        )

        similarity = raw_similarity

        # ====================================================
        # 14. FINAL MULTI-SIGNAL TRUST SCORE
        # ====================================================

        score_result = (
            calculate_final_trust_score(

                speaker_similarity=
                    similarity,

                naturalness_score=
                    naturalness_score,

                acoustic_score=
                    acoustic_score,

                conversation_risk=
                    conversation_risk,

                organization_match=True
            )
        )

        trust_score = (
            score_result[
                "final_trust_score"
            ]
        )

        # ====================================================
        # 15. TRUST LEVEL
        # ====================================================

        trust_level = (
            get_trust_level(

                trust_score,

                conversation_risk,

                similarity
            )
        )

        # ====================================================
        # 16. VERDICT
        # ====================================================
        # Identity is evaluated first. Conversation risk can override a
        # strong identity match when the interaction is dangerous.

        if conversation_risk >= 80:

            verdict = "HIGH_RISK_SCAM"
            message = (
                "The interaction contains critical scam indicators. "
                "Even if the voice matches the registered agent, "
                "the conversation itself is high risk."
            )

        elif similarity < 75:

            verdict = "VOICE_MISMATCH"
            message = (
                "The uploaded voice does not sufficiently match "
                "the registered reference voice for this organization."
            )

        elif conversation_risk >= 60:

            verdict = "SUSPICIOUS_CONVERSATION"
            message = (
                "The voice matches the registered organization agent, "
                "but the conversation contains significant risk indicators."
            )

        elif similarity < 86:

            verdict = "UNVERIFIED_VOICE"
            message = (
                "The organization was identified, but the voice match "
                "is not strong enough to verify the registered AI agent."
            )

        elif conversation_risk >= 30:

            verdict = "SUSPICIOUS_CONVERSATION"
            message = (
                "The registered AI agent voice matches strongly, "
                "but the conversation contains cautionary signals."
            )

        else:

            verdict = "LIKELY_VERIFIED"
            message = (
                f"The uploaded voice strongly matches the registered "
                f"{detected_organization} AI agent and no major "
                "conversation risk indicators were detected."
            )

        # 17. FINAL RESPONSE
        # ====================================================

        return {

            "success": True,

            "verdict": verdict,

            "trust_score": trust_score,

            "trust_level": trust_level,

            "similarity": similarity,

            "raw_similarity": raw_similarity,

            "naturalness_score":
                round(
                    naturalness_score,
                    2
                ),

            "acoustic_score":
                acoustic_score,

            "transcript": transcript,

            "organization_detected": True,

            "organization":
                detected_organization,

            "organization_id":
                detected_organization_id,

            "best_match":
                best_match,

            "all_matches":
                results,

            # =================================================
            # VOICE ANALYSIS
            # =================================================

            "voice_analysis": {

                "pitch_variation":
                    voice_analysis.get(
                        "pitch",
                        {}
                    ).get(
                        "pitch_variation",
                        0
                    ),

                "energy_variation":
                    voice_analysis.get(
                        "energy",
                        {}
                    ).get(
                        "energy_variation",
                        0
                    ),

                "pause_score":
                    voice_analysis.get(
                        "pauses",
                        {}
                    ).get(
                        "silence_ratio",
                        0
                    ),

                "spectral_score":
                    voice_analysis.get(
                        "analysis",
                        {}
                    ).get(
                        "spectral_consistency",
                        0
                    ),

                "speech_consistency":
                    voice_analysis.get(
                        "speech_consistency",
                        0
                    ),

                "naturalness_score":
                    naturalness_score,

                # Keep the original detailed analysis too.
                "details":
                    voice_analysis
            },

            # =================================================
            # RISK ANALYSIS
            # =================================================

            "risk_analysis": {

                "risk_score":
                    risk_analysis[
                        "risk_score"
                    ],

                "risk_level":
                    risk_analysis[
                        "risk_level"
                    ],

                "risk_flags":
                    risk_analysis[
                        "risk_flags"
                    ],

                "detected_signals":
                    risk_analysis[
                        "detected_signals"
                    ],

                "message":
                    risk_analysis[
                        "message"
                    ]
            },

            # =================================================
            # SCORE BREAKDOWN
            # =================================================

            "score_breakdown": {

                "speaker_similarity":
                    round(
                        similarity,
                        2
                    ),

                "speaker_similarity_weight":
                    100,

                "naturalness":
                    round(
                        naturalness_score,
                        2
                    ),

                "naturalness_weight":
                    0,

                "acoustic_score":
                    round(
                        acoustic_score,
                        2
                    ),

                "acoustic_weight":
                    0,

                "conversation_risk":
                    round(
                        conversation_risk,
                        2
                    ),

                "conversation_risk_weight":
                    0,

                "organization_match":
                    100,

                "organization_match_weight":
                    0,

                "supporting_audio_score":
                    score_result.get(
                        "supporting_audio_score",
                        0
                    ),

                "base_trust":
                    score_result[
                        "base_trust"
                    ],

                "risk_multiplier":
                    score_result[
                        "risk_multiplier"
                    ],

                "voice_mismatch_penalty":
                    score_result[
                        "mismatch_penalty"
                    ],

                "supporting_audio_score":
                    score_result.get(
                        "supporting_audio_score",
                        0
                    ),

                "final_trust_score":
                    trust_score
            },

            # =================================================
            # VERIFICATION
            # =================================================

            "verification": {

                "organization_match":
                    True,

                "voice_match":
                    similarity >= 86,

                "high_similarity":
                    similarity >= 90,

                "natural_speech":
                    naturalness_score >= 60,

                "conversation_low_risk":
                    conversation_risk < 30,

                "conversation_high_risk":
                    conversation_risk >= 60,

                "suspicious":
                    (
                        conversation_risk >= 30
                        or
                        similarity < 86
                    )
            },

            "message":
                message,

            "note": (
                "DeepShield combines speaker similarity, "
                "acoustic characteristics, naturalness, "
                "organization detection and contextual "
                "conversation-risk analysis. The "
                "conversation-risk score is an explainable "
                "hackathon MVP heuristic and is not a "
                "certified probability of fraud."
            )
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Voice verification failed: "
                f"{str(error)}"
            )
        )

    finally:

        # ====================================================
        # DELETE RAW AUDIO
        # ====================================================

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )


# ============================================================
# TRANSCRIBE UPLOADED VOICE
# ============================================================

@app.post("/api/voice/transcribe")
async def transcribe_voice(
    voice_file: UploadFile = File(...)
):

    temp_path = None

    allowed_types = [

        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mp4",
        "audio/x-m4a"
    ]

    if voice_file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid audio format. "
                "Upload MP3, WAV or M4A."
            )
        )

    audio_data = await voice_file.read()

    max_size = (
        20 * 1024 * 1024
    )

    if len(audio_data) == 0:

        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    if len(audio_data) > max_size:

        raise HTTPException(
            status_code=400,
            detail=(
                "Audio file must be "
                "smaller than 20 MB."
            )
        )

    try:

        extension = ".wav"

        if voice_file.filename:

            original_extension = (
                os.path.splitext(
                    voice_file.filename
                )[1]
                .lower()
            )

            if original_extension in [
                ".mp3",
                ".wav",
                ".m4a"
            ]:

                extension = (
                    original_extension
                )

        temp_filename = (
            f"transcribe_{uuid.uuid4().hex}"
            f"{extension}"
        )

        storage_directory = os.path.join(
            "storage",
            "voices"
        )

        os.makedirs(
            storage_directory,
            exist_ok=True
        )

        temp_path = os.path.join(
            storage_directory,
            temp_filename
        )

        with open(
            temp_path,
            "wb"
        ) as buffer:

            buffer.write(
                audio_data
            )

        transcript = transcribe_audio(
            temp_path
        )

        return {

            "success": True,

            "transcript": transcript,

            "message": (
                "Voice successfully "
                "converted to text."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Voice transcription failed: "
                f"{str(error)}"
            )
        )

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# DETECT ORGANIZATION FROM TRANSCRIPT
# ============================================================

@app.post("/api/detect-organization")
def detect_organization_from_transcript(

    transcript: str = Form(...),

    db: Session = Depends(get_db)
):

    organizations = (
        db.query(Organization)
        .all()
    )

    if not organizations:

        raise HTTPException(
            status_code=404,
            detail=(
                "No organizations are registered."
            )
        )

    result = detect_organization(
        transcript,
        organizations
    )

    if not result:

        return {

            "organization_detected": False,

            "organization": None,

            "message": (
                "No registered organization "
                "was detected in the transcript."
            )
        }

    return {

        "organization_detected": True,

        "organization":
            result["organization"],

        "domain":
            result["domain"],

        "organization_id":
            result["organization_id"],

        "matched_by":
            result["matched_by"],

        "message": (
            "Registered organization detected."
        )
    }