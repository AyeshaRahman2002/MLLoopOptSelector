# src/utils.py
import json, platform, shutil, subprocess, sys
from datetime import datetime
from typing import Dict, Any

def sysinfo() -> Dict[str, Any]:
    clang_ver = _safe_cmd(["clang", "--version"])
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "clang": clang_ver.strip().split("\n")[0] if clang_ver else "unknown",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

def _safe_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode()
    except Exception:
        return ""

def dump_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
