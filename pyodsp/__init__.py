import logging
from typing import IO

# A library must not configure logging on behalf of its host application.
# pyodsp only emits records, under the "pyodsp" logger tree, and attaches
# nothing but a NullHandler so that importing it stays silent until the
# application opts in -- either through configure_logging() below, or by
# wiring the "pyodsp" logger into its own logging setup.
logging.getLogger("pyodsp").addHandler(logging.NullHandler())

_DEFAULT_FORMAT = "%(levelname)s - %(pyodsp_prefix)s%(message)s"
_INDENT = "  "


class _PyodspFormatter(logging.Formatter):
    """Renders the decomposition-tree prefix from a record's structured context.

    A bundle-method solver tags each record with ``record.pyodsp`` -- its node
    id and depth. This turns that into a ``Node: <id> - `` prefix, indented by
    depth, exposed to the format string as ``%(pyodsp_prefix)s``. Records
    without the context (the algorithm-level loggers, anything third-party)
    get an empty prefix, so the format string is always valid.
    """

    def format(self, record: logging.LogRecord) -> str:
        context = getattr(record, "pyodsp", None)
        if isinstance(context, dict) and "node_id" in context:
            depth = context.get("depth") or 0
            indent = _INDENT * max(depth, 0)
            record.pyodsp_prefix = f"{indent}Node: {context['node_id']} - "
        else:
            record.pyodsp_prefix = ""
        return super().format(record)


def configure_logging(
    level: int = logging.INFO,
    *,
    stream: IO[str] | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Handler:
    """Send pyodsp's log output to the console.

    pyodsp follows the convention that a library does not configure logging;
    it only emits records under the ``pyodsp`` logger. Call this once from a
    script or application to watch the algorithms' progress without setting
    up ``logging`` yourself.

    ``level`` gates both the ``pyodsp`` logger and the handler this installs,
    so ``configure_logging(logging.WARNING)`` really does quiet the iteration
    output. Repeated calls replace the handler rather than stacking a second
    one. The handler is returned so it can be removed or restyled later.

    The handler renders a ``Node: <id> - `` prefix for the per-solver lines
    from the structured context each record carries (``%(pyodsp_prefix)s`` in
    ``fmt``); a custom ``fmt`` that omits it simply drops the prefix. For
    finer control, ignore this helper and configure the ``pyodsp`` logger (or
    a child such as ``pyodsp.dec.bd``) through the standard ``logging`` API --
    every record carries its numbers on ``record.pyodsp``.
    """
    logger = logging.getLogger("pyodsp")
    logger.setLevel(level)

    for existing in [h for h in logger.handlers if getattr(h, "_pyodsp", False)]:
        logger.removeHandler(existing)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(_PyodspFormatter(fmt))
    handler._pyodsp = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler


__all__ = ["configure_logging"]
