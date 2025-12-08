import logging
import os
from datetime import datetime

# Create logs folder if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,                           # Default log level
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()                   # Print logs on console
    ]
)

# Function to get a logger for any module
def get_logger(name: str = None):
    return logging.getLogger(name)


