# Handles Schwab auth + market data fetch
import os
import base64
import hashlib
import requests
from urllib.parse import urlencode

def generate_auth_url(client_id: str, redirect_uri: str, scope: str = "readonly"):
    """
    Generates the authorization URL for Schwab's OAuth 2.0 PKCE flow.

    Returns:
        (str, str): The authorization URL and the generated code_verifier
    """
    code_verifier = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b'=').decode('utf-8')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode('utf-8')).digest()
    ).rstrip(b'=').decode('utf-8')

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256"
    }

    auth_base = "https://api.schwabapi.com/v1/oauth/authorize"
    auth_url = f"{auth_base}?{urlencode(params)}"
    
    return auth_url, code_verifier


def exchange_code_for_tokens(auth_code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """
    Exchanges the authorization code for access and refresh tokens.

    Returns:
        dict: Contains access_token and refresh_token if successful.
    """
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri
    }

    response = requests.post("https://api.schwabapi.com/v1/oauth/token", headers=headers, data=payload)
    if response.ok:
        return response.json()
    else:
        raise Exception(f"Token exchange failed: {response.status_code} {response.text}")

