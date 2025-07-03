import psycopg2
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """Test database connection for health checks"""
    try:
        # Database connection parameters
        connection_params = {
            'host': os.getenv('DBHOST', 'localhost'),
            'port': os.getenv('DBPORT', '5432'),
            'database': os.getenv('DBNAME', 'postgres'),
            'user': os.getenv('DBUSER', 'postgres'),
            'password': os.getenv('DBPASSWORD', 'changeme')
        }
        
        # Test connection
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()
        
        # Simple query to test connection
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        logger.info("Database connection test successful")
        return True if result and result[0] == 1 else False
        
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False


def get_database_info():
    """Get database information for monitoring"""
    try:
        connection_params = {
            'host': os.getenv('DBHOST', 'localhost'),
            'port': os.getenv('DBPORT', '5432'),
            'database': os.getenv('DBNAME', 'postgres'),
            'user': os.getenv('DBUSER', 'postgres'),
            'password': os.getenv('DBPASSWORD', 'changeme')
        }
        
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()
        
        # Get database version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        
        # Get database size
        cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        size = cursor.fetchone()[0]
        
        # Get connection count
        cursor.execute("SELECT count(*) FROM pg_stat_activity")
        connections = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            "version": version,
            "size": size,
            "active_connections": connections,
            "status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Failed to get database info: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
