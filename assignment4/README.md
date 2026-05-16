# Assignment 4 — How to run

## Setup

```sh
uv venv --python /opt/homebrew/bin/python3.13
uv sync
```

(Homebrew Python is required because `mjpython` needs `libpython3.13.dylib`, which uv's `python-build-standalone` does not ship.)

## Exercises

```sh
uv run python exercise1.py
uv run python exercise2.py
.venv/bin/mjpython exercise3.py    # macOS passive viewer needs mjpython
```

Exercise 3 connects to the UR5e at `192.168.1.103`. The robot must be powered on, brakes released, and in **Remote Control** mode.

Jog keys: `w`/`s` = x, `a`/`d` = y, `q`/`e` = z, `Esc` = quit.
