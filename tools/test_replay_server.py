import http.client
import http.server
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_server import create_handler_class


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def remove_directory_recursive(target_dir: Path):
    if not target_dir.exists():
        return
    for item in target_dir.iterdir():
        if item.is_dir():
            remove_directory_recursive(item)
        else:
            item.unlink()
    target_dir.rmdir()


class TestReplayServerAPI(unittest.TestCase):
    server: http.server.ThreadingHTTPServer = None
    server_thread: threading.Thread = None
    server_port: int = 0
    base_url: str = ""
    test_replays_dir: Path = None
    repo_root: Path = None

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.test_replays_dir = cls.repo_root / "target" / "test_replays"
        remove_directory_recursive(cls.test_replays_dir)
        cls.test_replays_dir.mkdir(parents=True, exist_ok=True)

        folder_a = cls.test_replays_dir / "folder_a"
        folder_a.mkdir(parents=True, exist_ok=True)
        (folder_a / "match1.json").write_text(json.dumps({"game": "ants", "id": 1}))
        (folder_a / "match2.json").write_text(json.dumps({"game": "ants", "id": 2}))
        (folder_a / "notes.txt").write_text("not a replay")

        folder_b = cls.test_replays_dir / "folder_b"
        folder_b.mkdir(parents=True, exist_ok=True)
        (folder_b / "final.json").write_text(json.dumps({"game": "ants", "winner": 0}))

        empty_folder = cls.test_replays_dir / "empty_folder"
        empty_folder.mkdir(parents=True, exist_ok=True)

        hidden_folder = cls.test_replays_dir / ".hidden_folder"
        hidden_folder.mkdir(parents=True, exist_ok=True)
        (hidden_folder / "hidden.json").write_text(json.dumps({"game": "ants"}))

        viz_dir = cls.repo_root / "dist" / "viz"
        web_viewer_dir = cls.repo_root / "tools" / "web_viewer"

        handler_class = create_handler_class(cls.test_replays_dir, viz_dir, web_viewer_dir)
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
        cls.server_port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.server_port}"

        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.shutdown()
            cls.server.server_close()
        if cls.server_thread:
            cls.server_thread.join()
        remove_directory_recursive(cls.test_replays_dir)

    def fetch(self, path: str):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def test_directories_listing_and_counts(self):
        status, headers, body = self.fetch("/api/directories")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))

        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

        self.assertEqual(data[0]["folder"], "folder_a")
        self.assertEqual(data[0]["name"], "folder_a")
        self.assertEqual(data[0]["count"], 2)

        self.assertEqual(data[1]["folder"], "folder_b")
        self.assertEqual(data[1]["name"], "folder_b")
        self.assertEqual(data[1]["count"], 1)

    def test_matches_listing(self):
        status, headers, body = self.fetch("/api/matches?folder=folder_a")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))

        matches = json.loads(body.decode("utf-8"))
        self.assertEqual(matches, ["match1.json", "match2.json"])

    def test_matches_error_cases(self):
        status_missing, _, _ = self.fetch("/api/matches")
        self.assertEqual(status_missing, 400)

        status_empty, _, _ = self.fetch("/api/matches?folder=")
        self.assertEqual(status_empty, 400)

        status_notfound, _, _ = self.fetch("/api/matches?folder=nonexistent_folder")
        self.assertEqual(status_notfound, 404)

    def test_replay_payload_retrieval(self):
        status, headers, body = self.fetch("/api/replay?folder=folder_a&file=match1.json")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))

        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload, {"game": "ants", "id": 1})

    def test_replay_error_cases(self):
        status_missing_file, _, _ = self.fetch("/api/replay?folder=folder_a")
        self.assertEqual(status_missing_file, 400)

        status_missing_folder, _, _ = self.fetch("/api/replay?file=match1.json")
        self.assertEqual(status_missing_folder, 400)

        status_empty, _, _ = self.fetch("/api/replay?folder=&file=")
        self.assertEqual(status_empty, 400)

        status_invalid_ext, _, _ = self.fetch("/api/replay?folder=folder_a&file=notes.txt")
        self.assertEqual(status_invalid_ext, 400)

        status_notfound_file, _, _ = self.fetch("/api/replay?folder=folder_a&file=missing.json")
        self.assertEqual(status_notfound_file, 404)

        status_notfound_folder, _, _ = self.fetch("/api/replay?folder=missing&file=match1.json")
        self.assertEqual(status_notfound_folder, 404)

    def test_path_traversal_defense(self):
        traversal_query_cases = [
            "/api/matches?folder=..",
            "/api/matches?folder=../target",
            "/api/matches?folder=%2e%2e",
            "/api/matches?folder=/etc",
            "/api/matches?folder=.",
            "/api/replay?folder=..&file=match1.json",
            "/api/replay?folder=folder_a&file=../../etc/passwd",
            "/api/replay?folder=folder_a&file=/etc/passwd",
            "/api/replay?folder=folder_a&file=%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/api/replay?folder=/etc&file=passwd",
        ]
        for path in traversal_query_cases:
            status, _, _ = self.fetch(path)
            self.assertEqual(status, 403, f"Path traversal should return 403: {path}")

        conn = http.client.HTTPConnection("127.0.0.1", self.server_port)
        raw_traversal_paths = [
            "/cartridges/ants/../../tools/replay_server.py",
            "/cartridges/ants/%2e%2e/%2e%2e/tools/replay_server.py",
            "/../../etc/passwd",
            "/%2e%2e/%2e%2e/etc/passwd",
        ]
        for raw_path in raw_traversal_paths:
            conn.request("GET", raw_path)
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 403, f"Raw path traversal should return 403: {raw_path}")
        conn.close()

    def test_cartridge_static_routing(self):
        status_js, headers_js, body_js = self.fetch("/cartridges/ants/viz.js")
        self.assertEqual(status_js, 200)
        self.assertIn("application/javascript", headers_js.get("Content-Type", ""))
        self.assertTrue(len(body_js) > 0)

        status_wasm, headers_wasm, body_wasm = self.fetch("/cartridges/ants/engine/tb-ants.core.wasm")
        self.assertEqual(status_wasm, 200)
        self.assertIn("application/wasm", headers_wasm.get("Content-Type", ""))
        self.assertTrue(len(body_wasm) > 0)

        status_json, headers_json, _ = self.fetch("/cartridges/ants/engine.json")
        self.assertEqual(status_json, 200)
        self.assertIn("application/json", headers_json.get("Content-Type", ""))

        status_missing, _, _ = self.fetch("/cartridges/ants/missing_file.js")
        self.assertEqual(status_missing, 404)

    def test_web_viewer_assets(self):
        status_root, headers_root, body_root = self.fetch("/")
        self.assertEqual(status_root, 200)
        self.assertIn("text/html", headers_root.get("Content-Type", ""))
        self.assertIn(b'id="replay-console"', body_root)

        status_index, headers_index, body_index = self.fetch("/index.html")
        self.assertEqual(status_index, 200)
        self.assertIn("text/html", headers_index.get("Content-Type", ""))
        self.assertIn(b'id="replay-console"', body_index)

        status_js, headers_js, body_js = self.fetch("/app.js")
        self.assertEqual(status_js, 200)
        self.assertIn("application/javascript", headers_js.get("Content-Type", ""))
        self.assertIn(b"mount", body_js)

        status_css, headers_css, body_css = self.fetch("/style.css")
        self.assertEqual(status_css, 200)
        self.assertIn("text/css", headers_css.get("Content-Type", ""))
        self.assertIn(b"#replay-console", body_css)

        status_missing, _, _ = self.fetch("/nonexistent.html")
        self.assertEqual(status_missing, 404)


class TestServerProcessLifecycle(unittest.TestCase):
    def test_cli_startup_and_graceful_shutdown(self):
        repo_root = Path(__file__).resolve().parent.parent
        test_dir = repo_root / "target" / "test_lifecycle_replays"
        remove_directory_recursive(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

        sample_folder = test_dir / "sample"
        sample_folder.mkdir(parents=True, exist_ok=True)
        (sample_folder / "test.json").write_text(json.dumps({"ok": True}))

        free_port = find_free_port()
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "replay_server.py"),
            "--port",
            str(free_port),
            "--replays",
            str(test_dir),
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        url = f"http://127.0.0.1:{free_port}/api/directories"
        connected = False
        start_time = time.time()
        while time.time() - start_time < 5.0:
            try:
                with urllib.request.urlopen(url, timeout=0.5) as resp:
                    if resp.status == 200:
                        connected = True
                        break
            except Exception:
                time.sleep(0.1)

        self.assertTrue(connected, "Replay server did not respond within timeout period")

        proc.terminate()
        proc.wait(timeout=3.0)
        self.assertIsNotNone(proc.returncode)
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()

        remove_directory_recursive(test_dir)


if __name__ == "__main__":
    unittest.main()
