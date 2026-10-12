"""Isolated browser-test server using real graph/tools/checkpoints and scripted models."""

import argparse
import time
from tempfile import TemporaryDirectory

from test_collective import ScriptedModels

from kollektiv.webapp import ControlRoomServer, RunManager


class DemoModels(ScriptedModels):
    """Use a visible pace for README capture without changing graph or tool execution."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def structured(self, *args, **kwargs):
        time.sleep(self.delay)
        return super().structured(*args, **kwargs)

    def act(self, *args, **kwargs):
        time.sleep(self.delay)
        return super().act(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("delay must not be negative")
    with TemporaryDirectory(prefix="kollektiv-ui-") as directory:
        server = ControlRoomServer(
            ("127.0.0.1", args.port), RunManager(directory, models=DemoModels(args.delay))
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
