import logging
import subprocess
import sys


def test_main_file_handler_uses_utf8_encoding():
    """Verify that main.py configures tren-cli.log with utf-8 encoding."""
    code = (
        "import platforms.cli.main, logging, sys, pathlib; "
        "handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]; "
        "assert any(pathlib.Path(h.baseFilename).name == 'tren-cli.log' and h.encoding == 'utf-8' for h in handlers); "
        "sys.exit(0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Failed: {result.stderr}"


def test_main_file_handler_can_log_unicode_symbols(tmp_path):
    """Verify that FileHandler configured with utf-8 can write Unicode symbols without error."""
    log_file = tmp_path / "test_unicode.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    test_logger = logging.getLogger("test_unicode_logger")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    unicode_msg = "Status: ✓ Success ⚡️ Power ❌ Error 🎉 Complete"
    test_logger.info(unicode_msg)
    handler.flush()
    handler.close()

    content = log_file.read_text(encoding="utf-8")
    assert unicode_msg in content
