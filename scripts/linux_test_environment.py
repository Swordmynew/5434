"""Bootstrap an isolated Linux Spark test environment in the workspace.

Optional developer helper, not used on the configured GPU server. Download a
versioned uv wheel from PyPI, verify its published SHA256, and use its managed
Python rather than changing the Linux system Python or Java installation.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    tools = root / ".tools"
    tools.mkdir(exist_ok=True)
    uv = tools / "uv" / "uv"
    if not uv.exists():
        with urllib.request.urlopen("https://pypi.org/pypi/uv/0.6.14/json", timeout=60) as response:
            metadata = json.load(response)
        wheel = next(x for x in metadata["urls"] if x["filename"].endswith("manylinux_2_17_x86_64.manylinux2014_x86_64.whl"))
        destination = tools / wheel["filename"]
        urllib.request.urlretrieve(wheel["url"], destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != wheel["digests"]["sha256"]:
            raise ValueError("Downloaded uv wheel checksum mismatch")
        with zipfile.ZipFile(destination) as archive:
            member = next(n for n in archive.namelist() if n.endswith("/uv") and not n.endswith(".dist-info/uv"))
            uv.parent.mkdir(exist_ok=True)
            uv.write_bytes(archive.read(member))
        uv.chmod(0o755)
    environment = dict(__import__("os").environ, UV_PYTHON_INSTALL_DIR=str(tools / "python"))
    subprocess.run([str(uv), "venv", "--python", "3.12", str(root / ".venv-linux")], check=True, env=environment)
    subprocess.run([str(uv), "pip", "install", "--python", str(root / ".venv-linux/bin/python"),
                    "pyspark==3.5.8", "pytest==8.3.5", "jdk4py==17.0.9.2", "numpy==2.2.4", "pyarrow==19.0.1", "psutil==7.0.0", "setuptools==78.1.0"], check=True, env=environment)


if __name__ == "__main__":
    main()
