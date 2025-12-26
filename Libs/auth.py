"""
Authentication Module - Session-based login for admin routes.
"""

from functools import wraps
from flask import session, redirect, url_for, request, flash

# Default admin credentials (should be changed in production)
ADMIN_USERS = {
    'admin': 'admin123',
    'operator': 'operator123'
}


def login_required(f):
    """
    Decorator to protect routes that require authentication.
    Redirects to login page if user is not authenticated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            # Store the original URL to redirect after login
            session['next'] = request.url
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def check_credentials(username, password):
    """
    Verify username and password against stored credentials.
    Returns True if valid, False otherwise.
    """
    if username in ADMIN_USERS:
        return ADMIN_USERS[username] == password
    return False


def get_current_user():
    """Get the current logged-in username, or None if not logged in."""
    return session.get('user')


def is_logged_in():
    """Check if user is currently logged in."""
    return 'user' in session
