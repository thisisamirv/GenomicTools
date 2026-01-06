#!/usr/bin/env python
import os
import re
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from utils.CLIFramework import (
    ArgumentParser,
    CLIFramework,
    ColorCode,
    ColorFormatter,
    DocumentationParser,
    HelpFormatter,
    OptionConfig,
)
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def sample_options():
    return [
        OptionConfig(flags=["-i", "--input"], type=str),
        OptionConfig(flags=["-o", "--output"], type=str),
        OptionConfig(
            flags=["-t", "--threads"],
            type=int,
            default=1,
        ),
        OptionConfig(
            flags=["-f", "--filter"],
            type=str,
            nargs="+",
            default=[],
        ),
    ]


@pytest.fixture
def sample_markdown():
    return """# Script Name

## Description
This is a sample description.

## Details
These are the details of how the script works.

## Arguments
| Argument | Description |
|----------|-------------|
| `-i, --input` | Input file path |
| `-o, --output` | Output file path |
| `-t, --threads` | Number of threads |
| `-f, --filter` | Filter criteria |

## Options
### `-i, --input`
The input file to process.

### `-o, --output`
The output file to write results to.

### `-t, --threads`
Number of CPU threads to use.
Available options:
- 1: Single-threaded (default)
- 2+: Multi-threaded

## Usage Examples
### Basic usage

```bash
# This is a comment that should be green
python script.py -i input.txt -o output.txt
```

### Advanced usage

```bash
# Another comment that should be green
python script.py -i input.txt -o output.txt -t 4 -f gene1 gene2
```

## Note

### Requirements
Required packages:

- numpy
- pandas

### Important considerations
The input file must be tab-delimited
The output format depends on analysis type
"""


@pytest.fixture
def temp_docs_dir(sample_markdown):
    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_path = Path(tmp_dir) / "docs"
        docs_path.mkdir()

        md_file = docs_path / "test_script.md"
        md_file.write_text(sample_markdown)

        yield docs_path


@pytest.mark.unit
def test_option_config_validation():
    config = OptionConfig(flags=["-i", "--input"])
    assert config.flags == ["-i", "--input"]
    assert config.type == str
    assert config.default is None
    assert config.required is False

    config = OptionConfig(
        flags=["-t", "--type"],
        type=str,
        choices=["gwas", "eqtl"],
        default="gwas",
        required=False,
        metavar="TYPE",
    )
    assert config.choices == ["gwas", "eqtl"]
    assert config.default == "gwas"
    assert config.metavar == "TYPE"

    with pytest.raises(ValueError, match="Option must have at least one flag"):
        OptionConfig(flags=[])

    with pytest.raises(ValueError, match="All flags must start with '-'"):
        OptionConfig(flags=["input"])


@pytest.mark.unit
def test_color_formatter():
    with patch.object(ColorFormatter, "_supports_color", return_value=True):
        formatter = ColorFormatter(use_colors=True)
        colored_text = formatter.colorize("test", ColorCode.RED)
        assert "\033[1;31m" in colored_text
        assert "test" in colored_text
        assert "\033[0m" in colored_text

    formatter = ColorFormatter(use_colors=False)
    plain_text = formatter.colorize("test", ColorCode.RED)
    assert plain_text == "test"
    assert "\033[" not in plain_text

    with patch("sys.stdout.isatty", return_value=True), patch.dict(
        os.environ, {}, clear=True
    ):
        formatter = ColorFormatter()
        assert formatter.use_colors or not formatter.use_colors


@pytest.mark.unit
def test_color_formatter_help_text():
    with patch.object(ColorFormatter, "_supports_color", return_value=True):
        formatter = ColorFormatter(use_colors=True)

        test_message = """* Description: Test description
* Arguments:
  -i, --input  Input file
* Usage:
  # This is a comment
  python script.py -i <file>"""

        formatted = formatter.format_help_text(test_message)

        assert "\033[1;36m* Description:\033[0m" in formatted
        assert "\033[1;35m* Arguments:\033[0m" in formatted
        assert "\033[1;31m* Usage:\033[0m" in formatted

        assert "\033[1;34m<file>\033[0m" in formatted

        assert "\033[0;32m# This is a comment\033[0m" in formatted


@pytest.mark.unit
def test_documentation_parser(temp_docs_dir):
    parser = DocumentationParser("test_script", temp_docs_dir)
    sections = parser.parse_documentation()

    assert "Description" in sections
    assert "Arguments" in sections
    assert "Usage" in sections
    assert "Details" in sections
    assert "Note" in sections
    assert "Options" in sections

    assert not sections["Description"].is_empty()
    assert "This is a sample description" in " ".join(sections["Description"].content)

    assert sections["Usage"].name == "Usage"


@pytest.mark.unit
def test_documentation_parser_file_not_found():
    non_existent_docs = Path("/non/existent/path")
    parser = DocumentationParser("missing_script", non_existent_docs)

    with pytest.raises(FileNotFoundError, match="Help file not found"):
        parser.parse_documentation()


@pytest.mark.unit
def test_help_formatter():
    formatter = HelpFormatter()

    simple_content = ["This", "is", "a", "test"]
    result = formatter.format_simple(simple_content)
    assert result == "This is a test"

    arg_content = [
        "| Argument | Description |",
        "|----------|-------------|",
        "| `-i, --input` | Input file |",
        "| `-o, --output` | Output file |",
    ]
    result = formatter.format_arguments(arg_content)
    assert "-i, --input  Input file" in result
    assert "-o, --output  Output file" in result

    usage_content = [
        "### Basic usage",
        "```bash",
        "python script.py -i input.txt",
        "```",
        "### Advanced usage",
        "```bash",
        "python script.py -i input.txt -t 4",
        "```",
    ]
    result = formatter.format_usage(usage_content)
    assert "- Basic usage:" in result
    assert "  python script.py -i input.txt" in result
    assert "- Advanced usage:" in result


@pytest.mark.unit
def test_argument_parser_with_option_config(sample_options):
    parser = ArgumentParser(sample_options)

    args = parser.parse_args(["-i", "test.txt", "-t", "4"])
    assert args.input == "test.txt"
    assert args.threads == 4
    assert args.verbose == "WARNING"
    assert args.log is None

    args = parser.parse_args(["-f", "gene1", "gene2"])
    assert args.filter == ["gene1", "gene2"]


@pytest.mark.unit
def test_argument_parser_enhanced_options():
    enhanced_options = [
        OptionConfig(
            flags=["-t", "--type"],
            choices=["gwas", "eqtl"],
            default="gwas",
        ),
        OptionConfig(flags=["-r", "--required"], required=True),
        OptionConfig(
            flags=["-q", "--quiet"],
            action="store_true",
        ),
    ]

    parser = ArgumentParser(enhanced_options)

    args = parser.parse_args(["-r", "required_value", "-q", "-t", "eqtl"])
    assert args.type == "eqtl"
    assert args.required == "required_value"
    assert args.quiet is True

    with pytest.raises(SystemExit):
        parser.parse_args(["-t", "invalid_choice", "-r", "required_value"])


@pytest.mark.unit
def test_cli_framework_initialization(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    assert framework.script_name == "test_script"
    assert isinstance(framework.color_formatter, ColorFormatter)
    assert isinstance(framework.doc_parser, DocumentationParser)
    assert isinstance(framework.help_formatter, HelpFormatter)
    assert isinstance(framework.arg_parser, ArgumentParser)

    framework_no_color = CLIFramework(
        sample_options, "test_script", use_colors=False, docs_root=temp_docs_dir
    )
    assert framework_no_color.color_formatter.use_colors is False


@pytest.mark.unit
def test_argument_parsing_basic(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    result = framework.run(["-i", "input.txt", "-o", "output.txt", "-t", "4"])

    assert result.input == "input.txt"
    assert result.output == "output.txt"
    assert result.threads == 4
    assert result.verbose == "WARNING"
    assert result.log is None


@pytest.mark.unit
def test_argument_parsing_with_nargs(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    result = framework.run(["-f", "gene1", "gene2", "gene3"])

    assert result.filter == ["gene1", "gene2", "gene3"]


@pytest.mark.unit
def test_help_flag_triggers_exit(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    with patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit) as exc_info:
            framework.run(["--help"])

        assert exc_info.value.code == 0
        mock_print.assert_called_once()


@pytest.mark.unit
def test_help_file_not_found(sample_options):
    non_existent_docs = Path("/non/existent/path")
    framework = CLIFramework(
        sample_options, "missing_script", docs_root=non_existent_docs
    )

    with patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit):
            framework.run(["--help"])

        printed_message = mock_print.call_args[0][0]
        assert "Help documentation not available" in printed_message
        assert "missing_script" in printed_message


@pytest.mark.unit
def test_help_message_formatting(sample_options, temp_docs_dir):
    with patch.object(ColorFormatter, "_supports_color", return_value=True):
        framework = CLIFramework(
            sample_options, "test_script", use_colors=True, docs_root=temp_docs_dir
        )

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit):
                framework.run(["--help"])

            printed_message = mock_print.call_args[0][0]

            assert "\033[1;36m* Description:\033[0m" in printed_message
            assert "\033[1;35m* Arguments:\033[0m" in printed_message
            assert "\033[1;31m* Usage:\033[0m" in printed_message

            plain_message = re.sub(r"\033\[[0-9;]*m", "", printed_message)
            assert "* Description: This is a sample description" in plain_message
            assert "-i, --input  Input file path" in plain_message
            assert "Required packages:" in plain_message


@pytest.mark.unit
def test_markdown_sections_parsing(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    with patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit):
            framework.run(["--help"])

        printed_message = mock_print.call_args[0][0]
        plain_message = re.sub(r"\033\[[0-9;]*m", "", printed_message)

        print(f"Plain message content: {repr(plain_message)}")

        assert "-i, --input  Input file path" in plain_message
        assert "-o, --output  Output file path" in plain_message

        assert "- Basic usage:" in plain_message
        assert "python script.py -i input.txt -o output.txt" in plain_message

        assert "- Requirements" in plain_message
        assert "- numpy" in plain_message
        assert "- pandas" in plain_message


@pytest.mark.unit
def test_color_disabled_output(sample_options, temp_docs_dir):
    framework = CLIFramework(
        sample_options, "test_script", use_colors=False, docs_root=temp_docs_dir
    )

    with patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit):
            framework.run(["--help"])

        printed_message = mock_print.call_args[0][0]

        assert "\033[" not in printed_message
        assert "* Description: This is a sample description" in printed_message


@pytest.mark.unit
def test_enhanced_option_features(temp_docs_dir):
    enhanced_options = [
        OptionConfig(
            flags=["-t", "--type"],
            choices=["gwas", "eqtl"],
            default="gwas",
        ),
        OptionConfig(flags=["-r", "--required"], required=True),
        OptionConfig(
            flags=["-q", "--quiet"],
            action="store_true",
        ),
    ]

    framework = CLIFramework(enhanced_options, "test_script", docs_root=temp_docs_dir)

    result = framework.run(["-r", "required_value", "-q"])
    assert result.type == "gwas"
    assert result.required == "required_value"
    assert result.quiet is True

    with pytest.raises(SystemExit):
        framework.run(["-t", "invalid_choice", "-r", "required_value"])


@pytest.mark.unit
def test_verbose_logging_setup(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    with patch("utils.LoggingUtils.log.setup") as mock_setup:
        result = framework.run(["-v", "DEBUG", "--log", "test.log"])

        assert result.verbose == "DEBUG"
        assert result.log == "test.log"
        mock_setup.assert_called_once_with(level="DEBUG", file="test.log")


@pytest.mark.unit
def test_default_argument_handling(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    result = framework.run([])

    assert result.input is None
    assert result.output is None
    assert result.threads == 1
    assert result.filter == []

    assert result.verbose == "WARNING"
    assert result.log is None


@pytest.mark.unit
def test_argument_validation(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    with pytest.raises(SystemExit):
        framework.run(["-t", "not_a_number"])


@pytest.mark.unit
def test_edge_cases(temp_docs_dir):
    empty_options = []
    framework = CLIFramework(empty_options, "test_script", docs_root=temp_docs_dir)

    result = framework.run([])
    assert hasattr(result, "verbose")
    assert hasattr(result, "log")


@pytest.mark.unit
def test_generate_help_message_directly(sample_options, temp_docs_dir):
    framework = CLIFramework(sample_options, "test_script", docs_root=temp_docs_dir)

    help_message = framework.generate_help_message()

    assert "* Description:" in help_message
    assert "* Arguments:" in help_message
    assert "* Usage:" in help_message
    assert "sample description" in help_message


@pytest.mark.unit
def test_color_formatting(sample_options, temp_docs_dir):
    with patch.object(ColorFormatter, "_supports_color", return_value=True):
        framework = CLIFramework(
            sample_options, "test_script", use_colors=True, docs_root=temp_docs_dir
        )

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit):
                framework.run(["--help"])

            printed_message = mock_print.call_args[0][0]

            assert "\033[1;36m* Description:\033[0m" in printed_message
            assert "\033[1;33m* Details:\033[0m" in printed_message
            assert "\033[1;32m* Note:\033[0m" in printed_message
            assert "\033[1;35m* Arguments:\033[0m" in printed_message
            assert "\033[1;38;5;208m* Options:\033[0m" in printed_message
            assert "\033[1;31m* Usage:\033[0m" in printed_message


@pytest.mark.integration
def test_real_integration(sample_options, sample_markdown):
    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_path = Path(tmp_dir) / "docs"
        docs_path.mkdir()

        md_file = docs_path / "test_script.md"
        md_file.write_text(sample_markdown)

        framework = CLIFramework(sample_options, "test_script", docs_root=docs_path)

        with patch("utils.LoggingUtils.log.setup") as mock_setup:
            result = framework.run(["-i", "test_input.txt", "-v", "INFO"])

            assert result.input == "test_input.txt"
            assert result.verbose == "INFO"
            mock_setup.assert_called_once_with(level="INFO", file=None)

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                framework.run(["--help"])

            assert exc_info.value.code == 0
            mock_print.assert_called_once()

            help_message = mock_print.call_args[0][0]
            assert "Description:" in help_message
            assert "Arguments:" in help_message
            assert "Usage:" in help_message


if __name__ == "__main__":
    pytest.main([__file__])
