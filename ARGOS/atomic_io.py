"""Bounded recovery from Windows rename contention; never truncate the old result."""
import os
import time

RETRY_DELAYS = (.005, .010, .020, .040)
_replace_once = os.replace
_sleep = time.sleep


def replace_output(source, destination, stats):
    """Retry only Win32 access/sharing/lock errors; preserve both files on failure."""
    for attempt in range(len(RETRY_DELAYS)+1):
        try:
            _replace_once(source, destination)
            return
        except OSError as error:
            code = getattr(error, 'winerror', None)
            if code not in (5, 32, 33):
                raise
            stats.setdefault('io_errors', []).append(code)
            if attempt == len(RETRY_DELAYS):
                raise
            stats['io_retries'] = stats.get('io_retries', 0)+1
            start = time.perf_counter()
            _sleep(RETRY_DELAYS[attempt])
            stats['io_retry_wait_s'] = stats.get('io_retry_wait_s', 0.)+time.perf_counter()-start
