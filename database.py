import sqlite3
import hashlib
from datetime import datetime

class Database:
    def __init__(self, db_name='cdss_health_vault.db'):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_name, check_same_thread=False)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Health reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password):
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        
        cursor.execute(
            'SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?',
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        return user
    
    def save_report(self, user_id, category, symptoms, diagnosis):
        """Save health report"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO health_reports (user_id, category, symptoms, diagnosis) VALUES (?, ?, ?, ?)',
            (user_id, category, symptoms, diagnosis)
        )
        
        conn.commit()
        conn.close()
        return True
    
    def get_user_reports(self, user_id):
        """Get all reports for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM health_reports WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        
        reports = cursor.fetchall()
        conn.close()
        
        return reports
    
    def delete_report(self, report_id, user_id):
        """Delete a report (only if it belongs to the user)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM health_reports WHERE id = ? AND user_id = ?',
            (report_id, user_id)
        )
        
        conn.commit()
        conn.close()
        return True
    
    def get_report_by_id(self, report_id, user_id):
        """Get a specific report"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM health_reports WHERE id = ? AND user_id = ?',
            (report_id, user_id)
        )
        
        report = cursor.fetchone()
        conn.close()
        
        return report