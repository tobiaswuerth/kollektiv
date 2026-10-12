"""Mutation requests must name the listener, not a rebound browser hostname."""

import threading

import httpx
import pytest

from kollektiv.config import Settings
from kollektiv.webapp import ControlRoomServer, RunManager


@pytest.fixture
def origin_room(tmp_path):
    manager = RunManager(tmp_path)
    server = ControlRoomServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=5) as client:
        yield client, manager, server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("hostname", ["127.0.0.1", "localhost"])
def test_local_browser_mutations_are_allowed(origin_room, hostname):
    client, _, server = origin_room
    host = f"{hostname}:{server.server_port}"
    response = client.post(
        "/api/runs",
        json={"goal": "A local browser mission"},
        headers={"Host": host, "Origin": f"http://{host}", "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 201, response.text


def test_local_cli_mutations_without_origin_are_allowed(origin_room):
    client, _, _ = origin_room
    assert client.post("/api/runs", json={"goal": "A local CLI mission"}).status_code == 201


def test_matching_forged_host_and_origin_are_refused(origin_room):
    client, manager, server = origin_room
    host = f"attacker.example:{server.server_port}"
    response = client.post(
        "/api/runs",
        json={"goal": "Must not be created"},
        headers={"Host": host, "Origin": f"http://{host}", "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 403
    assert manager.list_runs() == []


@pytest.mark.parametrize("origin", ["https://127.0.0.1:{port}", "null", "http://localhost:1"])
def test_foreign_scheme_null_and_wrong_port_are_refused(origin_room, origin):
    client, manager, server = origin_room
    response = client.post(
        "/api/runs",
        json={"goal": "Must not be created"},
        headers={"Origin": origin.format(port=server.server_port)},
    )
    assert response.status_code == 403
    assert manager.list_runs() == []


def test_cross_site_fetch_metadata_is_refused(origin_room):
    client, manager, server = origin_room
    response = client.post(
        "/api/runs",
        json={"goal": "Must not be created"},
        headers={
            "Origin": f"http://127.0.0.1:{server.server_port}",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert response.status_code == 403
    assert manager.list_runs() == []


def test_delete_refuses_rebound_authority_and_accepts_local_origin(origin_room):
    client, manager, server = origin_room
    saved = manager.configurations.save({"name": "Protected team", "settings": Settings()})
    path = f"/api/configurations/{saved['id']}"
    forged = f"attacker.example:{server.server_port}"
    response = client.delete(
        path,
        headers={"Host": forged, "Origin": f"http://{forged}", "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 403
    assert manager.configurations.get(saved["id"]) == saved
    response = client.delete(
        path,
        headers={"Origin": f"http://127.0.0.1:{server.server_port}"},
    )
    assert response.status_code == 200
    assert manager.configurations.list() == []


@pytest.mark.parametrize(
    ("configured", "resolved", "extra_hosts"),
    [
        ("collective.lan", "192.0.2.10", {"collective.lan", "192.0.2.10"}),
        ("0.0.0.0", "0.0.0.0", set()),
    ],
)
def test_only_explicit_bound_hosts_are_added(
    monkeypatch, tmp_path, configured, resolved, extra_hosts
):
    # Avoid depending on an available LAN interface or a test machine's DNS.
    def bind(server, address, handler):
        server.server_address = (resolved, address[1])
        server.server_port = address[1]

    monkeypatch.setattr("kollektiv.webapp.ThreadingHTTPServer.__init__", bind)
    server = ControlRoomServer((configured, 8765), RunManager(tmp_path))
    assert server.allowed_authorities == {
        (hostname, 8765) for hostname in {"127.0.0.1", "localhost", "::1"} | extra_hosts
    }
