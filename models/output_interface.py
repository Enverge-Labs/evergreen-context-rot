import os
import sys
import threading
from datetime import datetime
from typing import Callable, Optional


OutputFunction = Callable[[str], None]


def stdout_output(message: str) -> None:
    print(message)


def marimo_output(mo) -> OutputFunction:
    """Factory function to hide the details of Marimo output and
    progress bar management.
    """

    class MarimoOutput:
        bar = None

        def __init__(self, mo):
            """Take an `mo` instance as arg, because this is the running notebook. Required."""
            self.mo = mo

        def __call__(self, message: str) -> None:
            self.mo.output.append(message)

        def start_progress(self, total: int) -> None:
            if self.bar is None:
                self.bar = self.mo.status.progress_bar(
                    total=total, remove_on_exit=False
                ).__enter__()

        def update(self) -> None:
            self.bar.update()

        def finish_progress(self) -> None:
            if self.bar is not None:
                self.bar.clear()
                self.bar.close()
                self.bar = None

    return MarimoOutput(mo)
