"""
RSR — Utility: Rate Limiting & Retry
Retry decorator with exponential backoff for API calls.
"""
import functools
import logging
import time

logger = logging.getLogger("rsr")


def retry(max_tries: int = 3, delay: float = 2.0, backoff: float = 2.0, exceptions=(Exception,)):
    """
    Decorator: retry a function on exception with exponential backoff.

    max_tries: total attempts
    delay:     initial wait in seconds
    backoff:   multiply delay by this on each retry
    exceptions: tuple of exception types to catch
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            last_exc = None
            for attempt in range(1, max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_tries:
                        logger.error(f"{func.__name__} failed after {max_tries} attempts: {e}")
                        raise
                    logger.warning(
                        f"{func.__name__}: attempt {attempt}/{max_tries} failed ({e}). "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    wait *= backoff
            raise last_exc
        return wrapper
    return decorator


def rate_limited(calls_per_minute: int = 60):
    """
    Decorator: ensure a function is not called more than N times per minute.
    Simple token-bucket approach.
    """
    min_interval = 60.0 / calls_per_minute
    last_called  = [0.0]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
