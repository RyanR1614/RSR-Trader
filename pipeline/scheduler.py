"""
RSR — Cloud Scheduler
Runs the daily pipeline on a schedule using the 'schedule' library.
Deploy this as the entry point on Railway, PythonAnywhere, or any always-on server.

Usage:
    python pipeline/scheduler.py
"""
import logging
import os
import sys
import time
from datetime import datetime

import schedule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOG_FILE

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rsr")


def job():
    """Wrapper that catches all exceptions so the scheduler keeps running."""
    logger.info(f"Scheduled job triggered at {datetime.utcnow().isoformat()}")
    try:
        from pipeline.daily_run import run_daily
        run_daily()
    except Exception as e:
        logger.error(f"Daily run failed: {e}", exc_info=True)


# ── Schedule: 9:35 AM ET on weekdays (14:35 UTC) ─────────────────────────────
# Adjust times here if needed. All times in UTC.
schedule.every().monday.at("14:35").do(job)
schedule.every().tuesday.at("14:35").do(job)
schedule.every().wednesday.at("14:35").do(job)
schedule.every().thursday.at("14:35").do(job)
schedule.every().friday.at("14:35").do(job)

logger.info("RSR Scheduler started.")
logger.info("Waiting for market open (14:35 UTC / 9:35 AM ET on weekdays)...")
logger.info(f"Next run: {schedule.next_run()}")

while True:
    schedule.run_pending()
    time.sleep(30)   # check every 30 seconds
