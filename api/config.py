"""Settings for the API + worker, all from environment variables so the
same code runs locally and inside a container without any code changes --
docker-compose.yml sets these per-service; running locally, the defaults
below assume Redis on localhost and a ./data folder in the current directory.
"""

from __future__ import annotations

import os
from pathlib import Path

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# book_path in a POST /process request is resolved relative to this
# directory (see worker.py). In docker-compose this is /data, bind-mounted
# from ./data on the host.
DATA_DIR = Path(os.environ.get("TIBETAN_DATA_DIR", "./data"))

# "cuda" or "cpu"; auto-detected in worker.py if not set here.
DEVICE = os.environ.get("TIBETAN_DEVICE")
