import http.client
import http.server
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_server import create_handler_class


class TestReplayServerAPI(unittest.TestCase):
    server: http.server.ThreadingHTTPServer = None
    server_thread: threading.Thread = None
    server_port: int = 0
    base_url: str = ""
    test_replays_dir: Path = None
    external_dir: Path = None
    locked_dir: Path = None
    locked_file: Path = None
    repo_root: Path = None

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.test_replays_dir = cls.repo_root / "target" / "test_replays"
        shutil.rmtree(cls.test_replays_dir, ignore_errors=True)
        cls.test_replays_dir.mkdir(parents=True, exist_ok=True)

        cls.external_dir = cls.repo_root / "target" / "test_external_dir"
        shutil.rmtree(cls.external_dir, ignore_errors=True)
        cls.external_dir.mkdir(parents=True, exist_ok=True)
        (cls.external_dir / "external_match.json").write_text(json.dumps({"external": True}))

        folder_a = cls.test_replays_dir / "folder_a"
        folder_a.mkdir(parents=True, exist_ok=True)
        (folder_a / "match1.json").write_text(json.dumps({"game": "ants", "id": 1}))
        (folder_a / "match2.json").write_text(json.dumps({"game": "ants", "id": 2}))
        (folder_a / "match.1..2.json").write_text(json.dumps({"game": "ants", "id": 3}))
        (folder_a / "win_50%.json").write_text(json.dumps({"game": "ants", "id": 4}))
        (folder_a / "notes.txt").write_text("not a replay")
        (folder_a / "outside_file_link.json").symlink_to(cls.external_dir / "external_match.json")

        # 权限 0o000 不可读文件
        cls.locked_file = folder_a / "locked.json"
        cls.locked_file.write_text(json.dumps({"game": "ants", "locked": True}))
        cls.locked_file.chmod(0o000)

        folder_b = cls.test_replays_dir / "folder_b"
        folder_b.mkdir(parents=True, exist_ok=True)
        (folder_b / "final.json").write_text(json.dumps({"game": "ants", "winner": 0}))

        # 内部跨文件夹符号链接：folder_a 下链接到 folder_b/final.json，应在 folder_a 中被过滤
        (folder_a / "cross_folder_link.json").symlink_to(folder_b / "final.json")

        empty_folder = cls.test_replays_dir / "empty_folder"
        empty_folder.mkdir(parents=True, exist_ok=True)

        hidden_folder = cls.test_replays_dir / ".hidden_folder"
        hidden_folder.mkdir(parents=True, exist_ok=True)
        (hidden_folder / "hidden.json").write_text(json.dumps({"game": "ants"}))

        (cls.test_replays_dir / "outside_folder_link").symlink_to(cls.external_dir, target_is_directory=True)

        # 权限 0o000 不可读目录
        cls.locked_dir = cls.test_replays_dir / "locked_dir"
        cls.locked_dir.mkdir(parents=True, exist_ok=True)
        (cls.locked_dir / "inside.json").write_text(json.dumps({"game": "ants", "secret": True}))
        cls.locked_dir.chmod(0o000)

        # 包含相互指向与自指向的符号链接环路目录
        loop_dir = cls.test_replays_dir / "loop_dir"
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / "valid.json").write_text(json.dumps({"game": "ants", "valid": True}))
        (loop_dir / "loop_a.json").symlink_to(loop_dir / "loop_b.json")
        (loop_dir / "loop_b.json").symlink_to(loop_dir / "loop_a.json")
        (loop_dir / "self_loop.json").symlink_to(loop_dir / "self_loop.json")

        # 检查真实构建产物 dist/viz 必须存在
        real_viz_dir = cls.repo_root / "dist" / "viz"
        if not real_viz_dir.exists() or not (real_viz_dir / "viz.js").exists():
            raise FileNotFoundError(
                f"Missing required build artifact directory: {real_viz_dir}. "
                "Please run './build.sh' and 'viz/build.sh' before running replay server tests."
            )
        viz_dir = real_viz_dir

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

        # 恢复权限后删除测试目录
        if hasattr(cls, "locked_dir") and cls.locked_dir is not None and cls.locked_dir.exists():
            try:
                cls.locked_dir.chmod(0o755)
            except OSError:
                pass
        if hasattr(cls, "locked_file") and cls.locked_file is not None and cls.locked_file.exists():
            try:
                cls.locked_file.chmod(0o644)
            except OSError:
                pass

        shutil.rmtree(cls.test_replays_dir, ignore_errors=True)
        if hasattr(cls, "external_dir") and cls.external_dir is not None:
            shutil.rmtree(cls.external_dir, ignore_errors=True)

    def fetch(self, path: str, method: str = "GET"):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method=method)
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

        dir_names = [d["name"] for d in data]
        self.assertNotIn("outside_folder_link", dir_names)
        self.assertNotIn(".hidden_folder", dir_names)
        self.assertNotIn("empty_folder", dir_names)
        self.assertNotIn("locked_dir", dir_names)

        self.assertIn("folder_a", dir_names)
        self.assertIn("folder_b", dir_names)
        self.assertIn("loop_dir", dir_names)

        folder_a_data = next(d for d in data if d["name"] == "folder_a")
        # 包含 match1.json, match2.json, match.1..2.json, win_50%.json, locked.json
        self.assertEqual(folder_a_data["count"], 5)

        folder_b_data = next(d for d in data if d["name"] == "folder_b")
        self.assertEqual(folder_b_data["count"], 1)

        loop_dir_data = next(d for d in data if d["name"] == "loop_dir")
        self.assertEqual(loop_dir_data["count"], 1)

    def test_matches_listing(self):
        status, headers, body = self.fetch("/api/matches?folder=folder_a")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))

        matches = json.loads(body.decode("utf-8"))
        self.assertEqual(matches, ["locked.json", "match.1..2.json", "match1.json", "match2.json", "win_50%.json"])
        self.assertNotIn("outside_file_link.json", matches)
        self.assertNotIn("cross_folder_link.json", matches)
        self.assertNotIn("notes.txt", matches)

    def test_matches_error_cases(self):
        status_missing, _, _ = self.fetch("/api/matches")
        self.assertEqual(status_missing, 400)

        status_empty, _, _ = self.fetch("/api/matches?folder=")
        self.assertEqual(status_empty, 400)

        status_notfound, _, _ = self.fetch("/api/matches?folder=nonexistent_folder")
        self.assertEqual(status_notfound, 404)

        # 嵌套斜杠必须被拦截为 403
        status_nested, _, _ = self.fetch("/api/matches?folder=folder_a/sub")
        self.assertEqual(status_nested, 403)

    def test_replay_payload_retrieval(self):
        status, headers, body = self.fetch("/api/replay?folder=folder_a&file=match1.json")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))

        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload, {"game": "ants", "id": 1})

        # 验证包含双点号与百分号的合法文件名能正常读取
        status_dot, _, body_dot = self.fetch("/api/replay?folder=folder_a&file=match.1..2.json")
        self.assertEqual(status_dot, 200)
        self.assertEqual(json.loads(body_dot.decode("utf-8")), {"game": "ants", "id": 3})

        status_pct, _, body_pct = self.fetch("/api/replay?folder=folder_a&file=win_50%.json")
        self.assertEqual(status_pct, 200)
        self.assertEqual(json.loads(body_pct.decode("utf-8")), {"game": "ants", "id": 4})

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

        # 跨文件夹软链接必须返回 403
        status_cross, _, _ = self.fetch("/api/replay?folder=folder_a&file=cross_folder_link.json")
        self.assertEqual(status_cross, 403)

    def test_unreadable_file_and_directory(self):
        # 0o000 权限不可读文件请求
        status_get, headers_get, body_get = self.fetch("/api/replay?folder=folder_a&file=locked.json")
        self.assertEqual(status_get, 403)
        self.assertIn("application/json", headers_get.get("Content-Type", ""))
        self.assertIn(b"Permission denied", body_get)

        status_head, headers_head, body_head = self.fetch(
            "/api/replay?folder=folder_a&file=locked.json", method="HEAD"
        )
        self.assertEqual(status_head, 403)
        self.assertEqual(len(body_head), 0)

        # 0o000 权限不可读目录请求
        status_matches, _, _ = self.fetch("/api/matches?folder=locked_dir")
        self.assertEqual(status_matches, 403)

        status_replay_locked_dir, _, _ = self.fetch("/api/replay?folder=locked_dir&file=inside.json")
        self.assertEqual(status_replay_locked_dir, 403)

    def test_symlink_loop_handling(self):
        # 包含循环软链接时 matches 接口应安全跳过循环链接
        status_matches, _, body_matches = self.fetch("/api/matches?folder=loop_dir")
        self.assertEqual(status_matches, 200)
        matches = json.loads(body_matches.decode("utf-8"))
        self.assertIn("valid.json", matches)
        self.assertNotIn("loop_a.json", matches)
        self.assertNotIn("loop_b.json", matches)
        self.assertNotIn("self_loop.json", matches)

        # 针对循环软链接文件的 replay 请求应返回 403
        status_replay, _, body_replay = self.fetch("/api/replay?folder=loop_dir&file=loop_a.json")
        self.assertEqual(status_replay, 403)
        self.assertIn(b"Forbidden: Path traversal detected", body_replay)

    def test_path_traversal_defense(self):
        traversal_query_cases = [
            "/api/matches?folder=..",
            "/api/matches?folder=../target",
            "/api/matches?folder=%2e%2e",
            "/api/matches?folder=/etc",
            "/api/matches?folder=.",
            "/api/matches?folder=....//",
            "/api/matches?folder=%252e%252e%252f",
            "/api/matches?folder=..\\\\",
            "/api/matches?folder=..\\\\..\\\\",
            "/api/matches?folder=outside_folder_link",
            "/api/matches?folder=%2e",
            "/api/matches?folder=%252e",
            "/api/replay?folder=..&file=match1.json",
            "/api/replay?folder=....//&file=match1.json",
            "/api/replay?folder=%252e%252e%252f&file=match1.json",
            "/api/replay?folder=..\\\\&file=match1.json",
            "/api/replay?folder=..\\\\..\\\\&file=match1.json",
            "/api/replay?folder=outside_folder_link&file=external_match.json",
            "/api/replay?folder=%2e&file=match1.json",
            "/api/replay?folder=%252e&file=match1.json",
            "/api/replay?folder=folder_a&file=../../etc/passwd",
            "/api/replay?folder=folder_a&file=/etc/passwd",
            "/api/replay?folder=folder_a&file=%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/api/replay?folder=folder_a&file=%252e%252e%252fetc%2fpasswd",
            "/api/replay?folder=folder_a&file=....//etc/passwd",
            "/api/replay?folder=folder_a&file=..\\\\etc/passwd",
            "/api/replay?folder=folder_a&file=outside_file_link.json",
            "/api/replay?folder=folder_a&file=%2e",
            "/api/replay?folder=folder_a&file=%252e",
            "/api/replay?folder=/etc&file=passwd",
        ]
        for path in traversal_query_cases:
            status, _, _ = self.fetch(path)
            self.assertEqual(status, 403, f"Path traversal should return 403: {path}")

        conn = http.client.HTTPConnection("127.0.0.1", self.server_port)
        try:
            raw_traversal_paths = [
                "/cartridges/ants/../../tools/replay_server.py",
                "/cartridges/ants/%2e%2e/%2e%2e/tools/replay_server.py",
                "/cartridges/ants/....//etc/passwd",
                "/cartridges/ants/%252e%252e%252fetc/passwd",
                "/cartridges/ants/..\\etc/passwd",
                "/cartridges/ants/..\\..\\etc/passwd",
                "/cartridges/ants/viz.js/.",
                "/cartridges/ants/%2e/viz.js",
                "/cartridges/ants/%252e/viz.js",
                "/../../etc/passwd",
                "/%2e%2e/%2e%2e/etc/passwd",
                "/....//etc/passwd",
                "/%252e%252e%252fetc/passwd",
                "/..\\etc/passwd",
                "/..\\..\\etc/passwd",
            ]
            for raw_path in raw_traversal_paths:
                conn.request("GET", raw_path)
                resp = conn.getresponse()
                resp.read()
                self.assertEqual(resp.status, 403, f"Raw path traversal should return 403: {raw_path}")
        finally:
            conn.close()

    def test_head_request_support(self):
        status, headers, body = self.fetch("/api/directories", method="HEAD")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))
        self.assertEqual(len(body), 0)

        status_root, headers_root, body_root = self.fetch("/", method="HEAD")
        self.assertEqual(status_root, 200)
        self.assertIn("text/html", headers_root.get("Content-Type", ""))
        self.assertEqual(len(body_root), 0)

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
        shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(test_dir, ignore_errors=True))

        sample_folder = test_dir / "sample"
        sample_folder.mkdir(parents=True, exist_ok=True)
        (sample_folder / "test.json").write_text(json.dumps({"ok": True}))

        real_viz_dir = repo_root / "dist" / "viz"
        if not real_viz_dir.exists() or not (real_viz_dir / "viz.js").exists():
            raise FileNotFoundError(
                f"Missing required build artifact directory: {real_viz_dir}. "
                "Please run './build.sh' and 'viz/build.sh' before running replay server tests."
            )

        cmd = [
            sys.executable,
            "-u",
            str(repo_root / "tools" / "replay_server.py"),
            "--port",
            "0",
            "--replays",
            str(test_dir),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def cleanup_process():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        self.addCleanup(cleanup_process)

        actual_port = None
        start_time = time.time()
        while time.time() - start_time < 5.0:
            if proc.poll() is not None:
                break
            line = proc.stdout.readline()
            if "listening on http://" in line:
                match = re.search(r":(\d+)/", line)
                if match:
                    actual_port = int(match.group(1))
                    break
            time.sleep(0.05)

        self.assertIsNotNone(actual_port, "Replay server did not output listening port.")

        url = f"http://127.0.0.1:{actual_port}/api/directories"
        connected = False
        start_time = time.time()
        while time.time() - start_time < 5.0:
            if proc.poll() is not None:
                break
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


if __name__ == "__main__":
    unittest.main()
