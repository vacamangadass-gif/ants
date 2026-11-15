import argparse
import http.server
import json
import mimetypes
from pathlib import Path
import typing
import urllib.parse

MIME_TYPES = {
    ".wasm": "application/wasm",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".map": "application/json",
}


class ReplayRequestHandler(http.server.BaseHTTPRequestHandler):
    replays_dir: Path = Path("replays")
    viz_dir: Path = Path("dist/viz")
    web_viewer_dir: Path = Path("tools/web_viewer")
    allowed_host: str = "127.0.0.1"

    def send_json(self, status_code: int, payload: typing.Any):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_file(self, file_path: Path, mime_type: str):
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def send_static_file(self, target_file: Path, default_mime: str = "application/octet-stream"):
        suffix = target_file.suffix.lower()
        mime = MIME_TYPES.get(suffix)
        if mime is None:
            guessed, _ = mimetypes.guess_type(target_file.name)
            mime = guessed or default_mime
        self.send_file(target_file, mime)

    def is_safe_subpath(self, base_dir: Path, *parts: str) -> tuple[bool, Path | None]:
        for part in parts:
            if not part or "\0" in part or "\\" in part:
                return False, None
            curr = part
            unquoted = urllib.parse.unquote(curr)
            while unquoted != curr:
                if "\0" in unquoted or "\\" in unquoted:
                    return False, None
                for seg in Path(unquoted).parts:
                    if seg == "." or (len(seg) >= 2 and all(c == "." for c in seg)):
                        return False, None
                curr = unquoted
                unquoted = urllib.parse.unquote(curr)
            p = Path(part)
            if p.is_absolute():
                return False, None
            for seg in p.parts:
                if seg == "." or (len(seg) >= 2 and all(c == "." for c in seg)):
                    return False, None
        combined = base_dir
        for part in parts:
            combined = combined / part
        try:
            resolved = combined.resolve()
            base_resolved = base_dir.resolve()
            if not resolved.is_relative_to(base_resolved):
                return False, None
            return True, resolved
        except (OSError, RuntimeError):
            return False, None

    def is_safe_flat_name(self, name: str) -> bool:
        if not name or "\0" in name or "/" in name or "\\" in name:
            return False
        curr = name
        unquoted = urllib.parse.unquote(curr)
        while unquoted != curr:
            if "\0" in unquoted or "/" in unquoted or "\\" in unquoted:
                return False
            for seg in Path(unquoted).parts:
                if seg == "." or (len(seg) >= 2 and all(c == "." for c in seg)):
                    return False
            curr = unquoted
            unquoted = urllib.parse.unquote(curr)
        for seg in Path(name).parts:
            if seg == "." or (len(seg) >= 2 and all(c == "." for c in seg)):
                return False
        if name.startswith("."):
            return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if path.startswith("/api/"):
            self.handle_api(path, query)
            return

        if path.startswith("/cartridges/ants/"):
            self.handle_cartridge(path)
            return

        self.handle_web_viewer(path)

    def handle_api(self, path: str, query: dict):
        if path == "/api/directories":
            dirs = []
            try:
                replays_resolved = self.replays_dir.resolve()
                for item in self.replays_dir.iterdir():
                    try:
                        if item.is_dir() and not item.name.startswith("."):
                            item_resolved = item.resolve()
                            if not item_resolved.is_relative_to(replays_resolved) or item_resolved == replays_resolved:
                                continue
                            count = 0
                            for f in item.iterdir():
                                try:
                                    if (
                                        f.is_file()
                                        and not f.name.startswith(".")
                                        and f.name.endswith(".json")
                                        and f.resolve().is_relative_to(item_resolved)
                                    ):
                                        count += 1
                                except (OSError, RuntimeError):
                                    continue
                            if count > 0:
                                dirs.append({"name": item.name, "folder": item.name, "count": count})
                    except (OSError, RuntimeError):
                        continue
            except (OSError, RuntimeError):
                self.send_json(500, {"error": "Failed to read replays directory"})
                return

            dirs.sort(key=lambda d: d["name"])
            self.send_json(200, dirs)
            return

        if path == "/api/matches":
            folder = query.get("folder", [None])[0]
            if folder is None or not folder.strip():
                self.send_json(400, {"error": "Missing required parameter: folder"})
                return

            if not self.is_safe_flat_name(folder):
                self.send_json(403, {"error": "Forbidden: Path traversal detected"})
                return

            safe, target_dir = self.is_safe_subpath(self.replays_dir, folder)
            if not safe or target_dir == self.replays_dir.resolve():
                self.send_json(403, {"error": "Forbidden: Path traversal detected"})
                return

            if not target_dir.exists() or not target_dir.is_dir():
                self.send_json(404, {"error": "Directory not found"})
                return

            matches = []
            try:
                target_resolved = target_dir.resolve()
                for f in target_dir.iterdir():
                    try:
                        if (
                            f.is_file()
                            and not f.name.startswith(".")
                            and f.name.endswith(".json")
                            and f.resolve().is_relative_to(target_resolved)
                        ):
                            matches.append(f.name)
                    except (OSError, RuntimeError):
                        continue
            except (OSError, RuntimeError):
                self.send_json(500, {"error": "Failed to read matches directory"})
                return

            matches.sort()
            self.send_json(200, matches)
            return

        if path == "/api/replay":
            folder = query.get("folder", [None])[0]
            file = query.get("file", [None])[0]
            if folder is None or not folder.strip() or file is None or not file.strip():
                self.send_json(400, {"error": "Missing required parameters: folder and file"})
                return

            if not self.is_safe_flat_name(folder) or not self.is_safe_flat_name(file):
                self.send_json(403, {"error": "Forbidden: Path traversal detected"})
                return

            if not file.endswith(".json"):
                self.send_json(400, {"error": "Invalid file extension: must be .json"})
                return

            safe_dir, target_dir = self.is_safe_subpath(self.replays_dir, folder)
            if not safe_dir or target_dir == self.replays_dir.resolve():
                self.send_json(403, {"error": "Forbidden: Path traversal detected"})
                return

            safe_file, target_file = self.is_safe_subpath(target_dir, file)
            if not safe_file:
                self.send_json(403, {"error": "Forbidden: Path traversal detected"})
                return

            if not target_dir.exists() or not target_dir.is_dir():
                self.send_json(404, {"error": "Directory not found"})
                return

            if not target_file.exists() or not target_file.is_file():
                self.send_json(404, {"error": "Replay file not found"})
                return

            self.send_file(target_file, "application/json; charset=utf-8")
            return

        self.send_json(404, {"error": "API endpoint not found"})

    def handle_cartridge(self, path: str):
        subpath = path[len("/cartridges/ants/"):].lstrip("/")
        if not subpath:
            self.send_json(404, {"error": "Static asset path missing"})
            return

        safe, target_file = self.is_safe_subpath(self.viz_dir, subpath)
        if not safe or target_file == self.viz_dir.resolve():
            self.send_json(403, {"error": "Forbidden: Path traversal detected"})
            return

        if not target_file.exists() or not target_file.is_file():
            self.send_json(404, {"error": "Static asset not found"})
            return

        self.send_static_file(target_file)

    def handle_web_viewer(self, path: str):
        if path == "/" or path == "/index.html":
            subpath = "index.html"
        else:
            subpath = path.lstrip("/")

        safe, target_file = self.is_safe_subpath(self.web_viewer_dir, subpath)
        if not safe or target_file == self.web_viewer_dir.resolve():
            self.send_json(403, {"error": "Forbidden: Path traversal detected"})
            return

        if not target_file.exists() or not target_file.is_file():
            self.send_json(404, {"error": "Page asset not found"})
            return

        self.send_static_file(target_file)


def create_handler_class(replays_dir: Path, viz_dir: Path, web_viewer_dir: Path):
    class ConfiguredHandler(ReplayRequestHandler):
        pass

    ConfiguredHandler.replays_dir = replays_dir.resolve()
    ConfiguredHandler.viz_dir = viz_dir.resolve()
    ConfiguredHandler.web_viewer_dir = web_viewer_dir.resolve()
    return ConfiguredHandler


def main():
    parser = argparse.ArgumentParser(description="Ants Replay Web Console Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--replays", type=str, default="", help="Path to replays directory (default: replays/)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.replays:
        replays_dir = Path(args.replays).resolve()
    else:
        replays_dir = (repo_root / "replays").resolve()

    if not replays_dir.exists() or not replays_dir.is_dir():
        raise FileNotFoundError(f"Replay directory does not exist: {replays_dir}")

    viz_dir = (repo_root / "dist" / "viz").resolve()
    if not viz_dir.exists() or not viz_dir.is_dir():
        raise FileNotFoundError(f"Visualization directory does not exist: {viz_dir}")

    web_viewer_dir = (repo_root / "tools" / "web_viewer").resolve()
    if not web_viewer_dir.exists() or not web_viewer_dir.is_dir():
        raise FileNotFoundError(f"Web viewer directory does not exist: {web_viewer_dir}")

    handler_class = create_handler_class(replays_dir, viz_dir, web_viewer_dir)
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler_class)
    server.daemon_threads = True

    print(f"Replay server listening on http://{args.host}:{args.port}/")
    print(f"Serving replays from: {replays_dir}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down replay server...")
        server.server_close()


if __name__ == "__main__":
    main()
