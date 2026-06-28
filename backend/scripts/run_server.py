import sys
import os
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    host = os.environ.get("MUTON_HOST", "0.0.0.0")
    port = int(os.environ.get("MUTON_PORT", "5000"))
    uvicorn.run("src.server:app", host=host, port=port, reload=False)
