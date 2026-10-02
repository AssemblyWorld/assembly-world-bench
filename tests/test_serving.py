"""Shared loopback delivery, large files and route isolation."""

import concurrent.futures
import hashlib
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import pytest

from assembly_world_bench.experiments.serving import EpisodeServer, environment_url


def test_large_files_shared_server_and_cleanup(tmp_path):
    path = tmp_path / "large.zip"
    with path.open("wb") as stream:
        for _ in range(51):
            stream.write(b"x" * 1024 * 1024)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with EpisodeServer("https://example.test/editor") as server:
        urls = [server.add(path) for _ in range(3)]
        assert len({urlsplit(url).netloc for url in urls}) == 1

        def download(url):
            digest = hashlib.sha256()
            with urlopen(Request(url, headers={"Origin": "https://example.test"})) as response:
                assert response.headers["Access-Control-Allow-Origin"] == "https://example.test"
                assert int(response.headers["Content-Length"]) == path.stat().st_size
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()

        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            assert list(pool.map(download, urls)) == [expected] * 3
        with urlopen(Request(urls[0], method="HEAD")) as response:
            assert response.read() == b""
        with urlopen(
            Request(
                urls[0],
                method="OPTIONS",
                headers={
                    "Origin": "https://example.test",
                    "Access-Control-Request-Private-Network": "true",
                },
            )
        ) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Private-Network"] == "true"
        for url, headers, status in [
            (urls[0], {"Origin": "https://other.test"}, 403),
            (urls[0] + "/../secret", {}, 404),
            (urls[0].rsplit("/", 2)[0] + "/", {}, 404),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url, headers=headers))
            assert error.value.code == status
        with pytest.raises(HTTPError) as error:
            urlopen(Request(urls[0], data=b"write", method="POST"))
        assert error.value.code == 501
    with pytest.raises(URLError):
        urlopen(urls[0], timeout=1)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_url_replaces_episode_and_preserves_environment_settings():
    url = environment_url(
        "https://example.test/editor?theme=dark&episode=old#view", "http://127.0.0.1:12/a.zip"
    )
    parsed = urlsplit(url)
    assert parsed.fragment == "view"
    assert parse_qs(parsed.query) == {"theme": ["dark"], "episode": ["http://127.0.0.1:12/a.zip"]}


def test_server_closes_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with EpisodeServer("https://example.test") as server:
            url = server.add(tmp_path / "missing.zip")
            with pytest.raises(HTTPError) as error:
                urlopen(url)
            assert error.value.code == 404
            assert error.value.headers["Access-Control-Allow-Origin"] == "https://example.test"
            raise RuntimeError("run failed")
    with pytest.raises(URLError):
        urlopen(url, timeout=1)
