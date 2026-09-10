from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey
)

import base64
import hashlib
import secrets


# ------------------------------------------------
# CRYPTOGRAPHIC IDENTITY
# ------------------------------------------------

def generate_identity():
    """
    Generate an Ed25519 public/private key pair.
    """

    private_key = Ed25519PrivateKey.generate()

    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes_raw()

    public_bytes = public_key.public_bytes_raw()

    private_key_b64 = base64.b64encode(
        private_bytes
    ).decode()

    public_key_b64 = base64.b64encode(
        public_bytes
    ).decode()

    return {
        "private_key": private_key_b64,
        "public_key": public_key_b64
    }


# ------------------------------------------------
# CHALLENGE SIGNING
# ------------------------------------------------

def sign_challenge(
    private_key_b64: str,
    challenge: str
):

    private_bytes = base64.b64decode(
        private_key_b64
    )

    private_key = Ed25519PrivateKey.from_private_bytes(
        private_bytes
    )

    signature = private_key.sign(
        challenge.encode()
    )

    return base64.b64encode(
        signature
    ).decode()


# ------------------------------------------------
# CHALLENGE VERIFICATION
# ------------------------------------------------

def verify_signature(
    public_key_b64: str,
    challenge: str,
    signature_b64: str
):

    try:

        public_bytes = base64.b64decode(
            public_key_b64
        )

        signature = base64.b64decode(
            signature_b64
        )

        public_key = Ed25519PublicKey.from_public_bytes(
            public_bytes
        )

        public_key.verify(
            signature,
            challenge.encode()
        )

        return True

    except Exception:

        return False


# ------------------------------------------------
# OTP
# ------------------------------------------------

def generate_otp():

    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str):

    return hashlib.sha256(
        otp.encode()
    ).hexdigest()


# ------------------------------------------------
# DOMAIN VERIFICATION TOKEN
# ------------------------------------------------

def generate_domain_token():

    return secrets.token_urlsafe(32)