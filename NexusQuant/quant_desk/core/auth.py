import os
import webbrowser
import urllib.parse
from typing import Optional
from fyers_apiv3 import fyersModel
from quant_desk.core.config import FYERS_CONFIG, update_access_token_in_env, get_access_token


def extract_auth_code_from_input(input_str: str) -> str:
    """Extract clean auth_code whether user pasted raw code or full redirect URL."""
    clean = input_str.strip()
    if "auth_code=" in clean or "http" in clean:
        try:
            parsed = urllib.parse.urlparse(clean)
            params = urllib.parse.parse_qs(parsed.query)
            if "auth_code" in params:
                return params["auth_code"][0]
            # Fallback if query string was passed directly
            if "?" in clean:
                query = clean.split("?", 1)[1]
                params = urllib.parse.parse_qs(query)
                if "auth_code" in params:
                    return params["auth_code"][0]
        except Exception:
            pass
    return clean


class FyersAuthenticator:
    """FYERS API v3 Authentication Manager."""

    def __init__(self):
        self.client_id = FYERS_CONFIG.client_id
        self.secret_key = FYERS_CONFIG.secret_key
        self.redirect_uri = FYERS_CONFIG.redirect_uri
        self.response_type = FYERS_CONFIG.response_type
        self.grant_type = FYERS_CONFIG.grant_type
        self.state = FYERS_CONFIG.state

    def generate_auth_code_url(self) -> str:
        """Generate the browser login URL to obtain auth_code."""
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type=self.response_type,
            grant_type=self.grant_type,
            state=self.state
        )
        auth_url = session.generate_authcode()
        return auth_url

    def open_browser_for_auth(self) -> str:
        """Generate auth URL and open it in user's default browser."""
        url = self.generate_auth_code_url()
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Auth] Could not open browser automatically: {e}")
        return url

    def generate_access_token(self, raw_input: str) -> str:
        """Exchange authorization code or redirect URL for an access token."""
        auth_code = extract_auth_code_from_input(raw_input)
        if not auth_code:
            raise ValueError("Invalid auth_code input.")

        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type=self.response_type,
            grant_type=self.grant_type,
            state=self.state
        )
        session.set_token(auth_code)
        response = session.generate_token()
        
        if isinstance(response, dict) and response.get("s") == "ok":
            access_token = response.get("access_token", "")
            if access_token:
                update_access_token_in_env(access_token)
                return access_token
            else:
                raise ValueError(f"Access token missing in response: {response}")
        else:
            raise RuntimeError(f"FYERS Token Generation Failed: {response}")

    def get_fyers_model(self, token: Optional[str] = None) -> fyersModel.FyersModel:
        """Initialize an active FYERS API instance."""
        access_token = token or get_access_token()
        if not access_token:
            raise ValueError("No FYERS Access Token found. Please authenticate first.")
        
        fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            is_async=False,
            token=access_token,
            log_path=os.path.join(os.getcwd(), "logs")
        )
        return fyers

    def validate_session(self, token: Optional[str] = None) -> bool:
        """Check if current token is valid by fetching user profile."""
        try:
            fyers = self.get_fyers_model(token)
            profile = fyers.get_profile()
            return isinstance(profile, dict) and profile.get("s") == "ok"
        except Exception:
            return False


if __name__ == "__main__":
    auth = FyersAuthenticator()
    print("=== FYERS OAuth2 Auth Generator ===")
    url = auth.open_browser_for_auth()
    print("1. Opening FYERS Login URL in your default browser:")
    print(url)
    inp = input("\n2. Paste the full redirect URL or the 'auth_code' parameter value here: ")
    if inp.strip():
        try:
            token = auth.generate_access_token(inp)
            print(f"\n[SUCCESS] FYERS Access Token generated & saved to .env: {token[:20]}...")
            if auth.validate_session(token):
                print("[SUCCESS] FYERS Session successfully validated with live profile!")
            else:
                print("[WARNING] Could not validate profile with generated token.")
        except Exception as err:
            print(f"\n[ERROR] Authentication failed: {err}")
