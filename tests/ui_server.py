"""Isolated browser-test server using real graph/tools/checkpoints and scripted models."""

from tempfile import TemporaryDirectory

from test_collective import ScriptedModels

from kollektiv.webapp import ControlRoomServer, RunManager

if __name__ == "__main__":
    with TemporaryDirectory(prefix="kollektiv-ui-") as directory:
        server = ControlRoomServer(
            ("127.0.0.1", 8767), RunManager(directory, models=ScriptedModels())
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
