"""Compute CRLF->LF normalized SHA-256 for the 128-file reference manifest."""
import hashlib
import sys
from pathlib import Path


def normalized_sha256(path: Path) -> str:
    data = path.read_bytes()
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main(reference: Path, out: Path) -> int:
    lines = reference.read_text(encoding="utf-8").splitlines()
    output_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        rel_path = parts[1]
        p = Path(rel_path)
        if not p.exists():
            output_lines.append(f"MISSING  {rel_path}")
            continue
        h = normalized_sha256(p)
        output_lines.append(f"{h}  {rel_path}")
    out.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(output_lines)} lines to {out}")
    return 0


if __name__ == "__main__":
    ref = Path(sys.argv[1])
    out = Path(sys.argv[2])
    sys.exit(main(ref, out))
