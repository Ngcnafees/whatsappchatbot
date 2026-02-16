import os, time
import httpx
import jwt

SF_LOGIN_URL = os.getenv("SF_LOGIN_URL", "https://test.salesforce.com").strip()
SF_CONSUMER_KEY = os.getenv("SF_CONSUMER_KEY", "").strip()
SF_USERNAME = os.getenv("SF_USERNAME", "").strip()
SF_PRIVATE_KEY_PATH = os.getenv("SF_PRIVATE_KEY_PATH", "").strip()

_access_token = None
_instance_url = None
_token_expires_at = 0  # unix time


def _load_private_key() -> str:
    if not SF_PRIVATE_KEY_PATH:
        raise RuntimeError("SF_PRIVATE_KEY_PATH is not set")
    with open(SF_PRIVATE_KEY_PATH, "r") as f:
        return f.read()


def _make_jwt_assertion() -> str:
    private_key = _load_private_key()
    now = int(time.time())
    payload = {
        "iss": SF_CONSUMER_KEY,
        "sub": SF_USERNAME,
        "aud": SF_LOGIN_URL,
        "exp": now + 180,  # 3 minutes
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


async def get_sf_token() -> tuple[str, str]:
    """
    Returns (access_token, instance_url). Caches token and refreshes when near expiry.
    """
    global _access_token, _instance_url, _token_expires_at

    # Reuse token if still valid for at least 2 minutes
    if _access_token and time.time() < (_token_expires_at - 120):
        return _access_token, _instance_url

    assertion = _make_jwt_assertion()
    url = f"{SF_LOGIN_URL}/services/oauth2/token"
    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        j = resp.json()

    _access_token = j["access_token"]
    _instance_url = j["instance_url"]
    # Salesforce doesn’t always return expires_in for JWT flow reliably; assume ~1 hour cache
    _token_expires_at = time.time() + 3300  # 55 minutes

    return _access_token, _instance_url
