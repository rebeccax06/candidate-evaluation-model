"""Supabase authentication for candidate evaluator."""

import streamlit as st
from streamlit import session_state as ss
from typing import Optional
from supabase import Client
from datetime import datetime, timedelta
import time

from candidate_evaluator.database import get_supabase_client, Database

from streamlit_cookies_controller import CookieController


# Cookie controller must be created at module level
controller = CookieController(key='cookies')

# Cookie prefix for this app
COOKIE_PREFIX = "eval_app"


def get_cookie_controller():
    """Get the cookie controller instance."""
    return controller


def init_auth_state():
    """Initialize authentication state in Streamlit session."""
    # Check if this is a fresh page load (not already initialized this session)
    if "auth_initialized" not in ss:
        ss.auth_initialized = True
        ss.user = None
        ss.supabase_client = None
        ss.db = None
        
        # Give cookies time to load from JavaScript
        cookies = controller.getAll()
        time.sleep(1)
        
        # Try to restore session from cookies using .get() for individual values
        cookie_user_id = controller.get(f'{COOKIE_PREFIX}_user_id')
        cookie_user_email = controller.get(f'{COOKIE_PREFIX}_user_email')
        cookie_access_token = controller.get(f'{COOKIE_PREFIX}_access_token')
        cookie_refresh_token = controller.get(f'{COOKIE_PREFIX}_refresh_token')
        
        if cookie_user_id and cookie_user_email:
            ss.user = {
                "id": cookie_user_id,
                "email": cookie_user_email,
                "created_at": ""
            }
            
            # Restore authenticated Supabase session so storage/db use the user JWT
            if cookie_access_token and cookie_refresh_token:
                try:
                    client = get_supabase_client()
                    client.auth.set_session(cookie_access_token, cookie_refresh_token)
                    # Explicitly propagate token to storage (Streamlit callbacks unreliable)
                    client.storage.set_auth(cookie_access_token)
                    ss.supabase_client = client
                    ss.access_token = cookie_access_token
                except Exception:
                    # Token may be expired; user will need to re-sign in
                    ss.supabase_client = None
            
            st.toast(f"Welcome back, {cookie_user_email}!")
    
    return True




def get_auth_client() -> Client:
    """Get or create Supabase client for authentication."""
    if st.session_state.supabase_client is None:
        st.session_state.supabase_client = get_supabase_client()
    return st.session_state.supabase_client


def get_database() -> Database:
    """Get or create Database instance."""
    if st.session_state.db is None:
        st.session_state.db = Database(get_auth_client())
    return st.session_state.db


def get_current_user() -> Optional[dict]:
    """Get currently logged in user."""
    return st.session_state.get("user")


def is_logged_in() -> bool:
    """Check if user is logged in."""
    return st.session_state.get("user") is not None


def sign_up(email: str, password: str) -> tuple[bool, str]:
    """
    Sign up a new user.
    
    Returns (success, message).
    """
    try:
        client = get_auth_client()
        response = client.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user:
            return True, "Account created! Please check your email to verify your account."
        else:
            return False, "Failed to create account. Please try again."
    
    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower():
            return False, "This email is already registered. Please sign in instead."
        return False, f"Error: {error_msg}"


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """
    Sign in an existing user.
    
    Returns (success, message).
    """
    try:
        client = get_auth_client()
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            ss.user = {
                "id": response.user.id,
                "email": response.user.email,
                "created_at": str(response.user.created_at)
            }
            
            # Explicitly propagate token to storage (Streamlit callbacks unreliable)
            if response.session:
                try:
                    client.storage.set_auth(response.session.access_token)
                    ss.access_token = response.session.access_token
                except Exception:
                    pass
            
            # Save to cookies for persistence (include session tokens)
            try:
                controller.set(f"{COOKIE_PREFIX}_user_id", response.user.id)
                controller.set(f"{COOKIE_PREFIX}_user_email", response.user.email)
                if response.session:
                    controller.set(f"{COOKIE_PREFIX}_access_token", response.session.access_token)
                    controller.set(f"{COOKIE_PREFIX}_refresh_token", response.session.refresh_token)
            except Exception as e:
                st.warning(f"Could not save session cookie: {e}")
            
            return True, "Signed in successfully!"
        else:
            return False, "Invalid email or password."
    
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower():
            return False, "Invalid email or password."
        return False, f"Error: {error_msg}"


def sign_out() -> tuple[bool, str]:
    """
    Sign out the current user.
    
    Returns (success, message).
    """
    try:
        client = get_auth_client()
        client.auth.sign_out()
        
        ss.user = None
        ss.supabase_client = None
        ss.db = None
        
        # Clear cookies
        try:
            controller.remove(f"{COOKIE_PREFIX}_user_id")
            controller.remove(f"{COOKIE_PREFIX}_user_email")
            controller.remove(f"{COOKIE_PREFIX}_access_token")
            controller.remove(f"{COOKIE_PREFIX}_refresh_token")
        except Exception:
            pass
        
        return True, "Signed out successfully."
    
    except Exception as e:
        ss.user = None
        return True, "Signed out."


def reset_password(email: str) -> tuple[bool, str]:
    """
    Send password reset email.
    
    Returns (success, message).
    """
    try:
        client = get_auth_client()
        client.auth.reset_password_email(email)
        return True, "Password reset email sent. Please check your inbox."
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def render_auth_ui() -> bool:
    """
    Render authentication UI.
    
    Returns True if user is authenticated, False otherwise.
    """
    init_auth_state()
    
    if is_logged_in():
        return True
    
    st.title("Candidate Evaluator")
    st.markdown("AI-powered candidate evaluation")
    
    st.markdown("---")
    
    # Check if Supabase is configured before showing auth forms
    try:
        get_auth_client()
    except Exception as e:
        st.error("Supabase is not configured properly.")
        st.code(str(e))
        st.markdown("""
        **To fix this:**
        1. Go to your Streamlit Cloud app settings
        2. Click on "Secrets"
        3. Add the following:
        ```
        SUPABASE_URL = "https://your-project.supabase.co"
        SUPABASE_KEY = "your-anon-public-key"
        ```
        4. Get these values from Supabase Dashboard → Settings → API
        """)
        return False
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        st.markdown("### Sign In")
        
        with st.form("signin_form"):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    success, message = sign_in(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        if st.button("Forgot password?", key="forgot_password"):
            st.session_state.show_reset = True
        
        if st.session_state.get("show_reset"):
            with st.form("reset_form"):
                reset_email = st.text_input("Enter your email", key="reset_email")
                if st.form_submit_button("Send Reset Link"):
                    if reset_email:
                        success, message = reset_password(reset_email)
                        if success:
                            st.success(message)
                            st.session_state.show_reset = False
                        else:
                            st.error(message)
    
    with tab2:
        st.markdown("### Create Account")
        
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            st.caption("Password must be at least 6 characters.")
            
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not new_email or not new_password:
                    st.error("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, message = sign_up(new_email, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    return False


def render_user_menu():
    """Render user menu in sidebar."""
    user = get_current_user()
    if not user:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{user['email']}**")
    
    if st.sidebar.button("Sign Out", key="signout_btn"):
        sign_out()
        st.rerun()


def require_api_key() -> Optional[str]:
    """
    Ensure user has an Anthropic API key set.
    
    Returns the API key if available, None otherwise.
    Renders UI to set the key if not available.
    """
    user = get_current_user()
    if not user:
        return None
    
    db = get_database()
    api_key = db.get_user_api_key(user["id"])
    
    if api_key:
        return api_key
    
    st.warning("Please set your Anthropic API key to use the evaluator.")
    st.markdown("You can get an API key from [Anthropic Console](https://console.anthropic.com/)")
    
    with st.form("api_key_form"):
        new_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-..."
        )
        
        if st.form_submit_button("Save API Key"):
            if new_key and new_key.startswith("sk-ant-"):
                db.update_user_api_key(user["id"], new_key)
                st.success("API key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid Anthropic API key (starts with sk-ant-)")
    
    return None
