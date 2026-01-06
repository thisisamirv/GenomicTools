#!/usr/bin/env python
# Import required modules
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .LoggingUtils import log


class ColorCode(Enum):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    MAGENTA = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    BRIGHT_CYAN = "\033[1;96m"
    BRIGHT_MAGENTA = "\033[1;95m"
    ORANGE = "\033[1;38;5;208m"
    DARK_GREEN = "\033[0;32m"


@dataclass
class OptionConfig:
    flags: List[str]
    type: type = str
    default: Any = None
    nargs: Optional[Union[int, str]] = None
    choices: Optional[List[str]] = None
    required: bool = False
    metavar: Optional[str] = None
    action: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.flags:
            raise ValueError("Option must have at least one flag")
        if not all(flag.startswith("-") for flag in self.flags):
            raise ValueError("All flags must start with '-'")


@dataclass
class HelpSection:
    name: str
    content: List[str] = field(default_factory=list)
    formatter: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.content or not any(line.strip() for line in self.content)


class ColorFormatter:
    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self) -> bool:
        """Check if the terminal supports color output."""
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        no_color_env = os.getenv("NO_COLOR") is None
        return is_tty and no_color_env

    def colorize(self, text: str, color: ColorCode) -> str:
        """Wrap text with ANSI color codes if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{color.value}{text}{ColorCode.RESET.value}"

    def format_help_text(self, message: str) -> str:
        """Apply color formatting to help message headers and special elements."""
        if not self.use_colors:
            return message
        header_colors = {
            "Description": ColorCode.CYAN,
            "Details": ColorCode.YELLOW,
            "Note": ColorCode.GREEN,
            "Input Format": ColorCode.BRIGHT_MAGENTA,
            "Output Format": ColorCode.BRIGHT_CYAN,
            "Arguments": ColorCode.MAGENTA,
            "Options": ColorCode.ORANGE,
            "Usage": ColorCode.RED,
        }
        for header, color in header_colors.items():
            message = re.sub(
                rf"(^\* {header}:)",
                f"{color.value}\\1{ColorCode.RESET.value}",
                message,
                flags=re.MULTILINE,
            )
        message = re.sub(
            r"<([^>]+)>", f"{ColorCode.BLUE.value}<\\1>{ColorCode.RESET.value}", message
        )
        message = re.sub(
            r"^(\s+)(#.*)$",
            f"\\1{ColorCode.DARK_GREEN.value}\\2{ColorCode.RESET.value}",
            message,
            flags=re.MULTILINE,
        )
        return message


class DocumentationParser:
    SECTION_MAPPING = {"Usage Examples": "Usage"}

    def __init__(self, script_name: str, docs_root: Optional[Path] = None) -> None:
        self.script_name = script_name
        self.docs_root = docs_root or self._find_docs_root()

    def _find_docs_root(self) -> Path:
        """Locate the 'docs/api' directory relative to this script."""
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent.parent
        return root_dir / "docs" / "api"

    def _get_help_file_path(self) -> Path:
        """Construct the path to the help documentation file."""
        return self.docs_root / f"{self.script_name}.md"

    def parse_documentation(self) -> Dict[str, HelpSection]:
        """Parse the markdown help documentation into structured sections."""
        help_file = self._get_help_file_path()
        if not help_file.exists():
            raise FileNotFoundError(f"Help file not found: {help_file}")
        with help_file.open("r", encoding="utf-8") as file:
            lines = [line.rstrip() for line in file.readlines()]
        sections = self._initialize_sections()
        current_section = None
        for line in lines:
            if line.startswith("## "):
                section_name = line[3:].strip()
                section_name = self.SECTION_MAPPING.get(section_name, section_name)
                current_section = section_name if section_name in sections else None
            elif current_section and (line.startswith("### ") or line.strip()):
                sections[current_section].content.append(line)
        return sections

    def _initialize_sections(self) -> Dict[str, HelpSection]:
        """Initialize the help sections with their respective formatters."""
        return {
            "Description": HelpSection("Description", formatter="format_simple"),
            "Details": HelpSection("Details", formatter="format_simple"),
            "Note": HelpSection("Note", formatter="format_note"),
            "Input Format": HelpSection(
                "Input Format", formatter="format_format_section"
            ),
            "Output Format": HelpSection(
                "Output Format", formatter="format_format_section"
            ),
            "Arguments": HelpSection("Arguments", formatter="format_arguments"),
            "Options": HelpSection("Options", formatter="format_options"),
            "Usage": HelpSection("Usage", formatter="format_usage"),
        }


class HelpFormatter:
    def format_simple(self, lines: List[str]) -> str:
        """Format simple sections by joining non-empty lines."""
        return " ".join(line.strip() for line in lines if line.strip())

    def format_arguments(self, lines: List[str]) -> str:
        """Format the arguments section."""
        arg_lines = [
            line for line in lines[2:] if line.strip() and not line.startswith("##")
        ]
        args: List[str] = []
        for line in arg_lines:
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if len(parts) >= 2:
                arg, desc = parts[0].replace("`", ""), parts[1]
                args.append(f"  {arg}  {desc}")
        return "\n".join(args)

    def format_usage(self, lines: List[str]) -> str:
        """Format the usage section."""
        formatted_usage: List[str] = []
        in_code_block = False
        for line in lines:
            line = line.strip()
            if line.startswith("### "):
                if formatted_usage:
                    formatted_usage.append("")
                header = line[4:].strip()
                formatted_usage.append(f"- {header}:")
            elif line.startswith("```"):
                in_code_block = not in_code_block
            elif in_code_block and line:
                formatted_usage.append(f"  {line}")
        return "\n".join(formatted_usage)

    def format_format_section(self, lines: List[str]) -> str:
        """Format input/output format sections."""
        formatted: List[str] = []
        in_code_block = False
        for line in lines:
            line = line.strip()
            if line.startswith("### "):
                if formatted:
                    formatted.append("")
                header = line[4:].strip()
                formatted.append(f"- {header}:")
            elif line.startswith("```"):
                in_code_block = not in_code_block
            elif in_code_block and line:
                formatted.append(f"  {line}")
            elif line and not line.startswith("#"):
                formatted.append(f"  {line}")
        return "\n".join(formatted)

    def format_options(self, lines: List[str]) -> str:
        """Format the options section."""
        if not lines:
            return ""
        options = self._parse_options(lines)
        formatted_options: List[str] = []
        for i, opt in enumerate(options):
            if i > 0:
                formatted_options.append("")
            formatted_options.append(opt["name"])
            description = self._format_option_description(opt["description"])
            if description:
                formatted_options.append(description)
        return "\n".join(formatted_options)

    def _parse_options(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse options from the lines into structured dictionaries."""
        options: List[Dict[str, Any]] = []
        current_option: Optional[Dict[str, Any]] = None
        for line in lines:
            line = line.strip()
            if line.startswith("### "):
                if current_option:
                    options.append(current_option)
                option_name = re.sub(r"`([^`]+)`", r"\1", line[4:].strip())
                current_option = {"name": option_name, "description": []}
            elif current_option and line:
                current_option["description"].append(line)
        if current_option:
            options.append(current_option)
        return options

    def _format_option_description(self, description_lines: List[str]) -> str:
        """Format the description of an option."""
        formatted: List[str] = []
        for line in description_lines:
            if "Available options:" in line:
                formatted.append(line.strip())
            elif line.startswith("- "):
                item = line[2:].replace("**", "").strip()
                formatted.append(f"  - {item}")
            else:
                formatted.append(line.strip())
        return "\n".join(formatted)

    def format_note(self, lines: List[str]) -> str:
        """Format the note section."""
        formatted: List[str] = []
        previous_line_empty = False
        for line in lines:
            line = line.strip()
            if line.startswith("### "):
                if formatted and not previous_line_empty:
                    formatted.append("")
                section = line[4:].strip()
                formatted.append(f"- {section}")
                previous_line_empty = False
            elif "**" in line:
                if formatted and not previous_line_empty:
                    formatted.append("")
                formatted.append(line.replace("**", ""))
                previous_line_empty = False
            elif re.match(r"^\d+\.", line):
                formatted.append(f"  {line}")
                previous_line_empty = False
            elif line.startswith("- "):
                formatted.append(f"  {line}")
                previous_line_empty = False
            elif line and not line.startswith("```"):
                if formatted and not previous_line_empty:
                    formatted.append("")
                formatted.append(line)
                previous_line_empty = False
        return "\n".join(formatted)


class ArgumentParser:
    def __init__(self, option_list: List[Union[Dict[str, Any], OptionConfig]]) -> None:
        self.options = self._normalize_options(option_list)
        self._check_conflicts()
        self.parser = self._create_parser()

    def _check_conflicts(self) -> None:
        """Ensure no option flags conflict with framework-reserved flags."""
        framework_flags = {"-l", "--log", "-v", "--verbose"}
        for opt in self.options:
            for flag in opt.flags:
                if flag in framework_flags:
                    raise ValueError(
                        f"Option flag '{flag}' conflicts with framework option. "
                        f"Framework reserves: {sorted(framework_flags)}"
                    )

    def _normalize_options(
        self, option_list: List[Union[Dict[str, Any], OptionConfig]]
    ) -> List[OptionConfig]:
        """Convert all options to OptionConfig instances."""
        normalized: List[OptionConfig] = []
        for opt in option_list:
            if isinstance(opt, dict):
                normalized.append(OptionConfig(**opt))
            elif isinstance(opt, OptionConfig):
                normalized.append(opt)
            else:
                raise ValueError(f"Invalid option type: {type(opt)}")
        return normalized

    def str_to_bool(self, value: Any) -> bool:
        """Convert common string representations to boolean for argparse."""
        if isinstance(value, bool):
            return value
        if value is None:
            raise argparse.ArgumentTypeError("Boolean value expected, got None")
        s = str(value).strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0", ""):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected; got: {value}")

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        parser = argparse.ArgumentParser(add_help=False)
        for opt in self.options:
            if opt.type is bool and opt.action is None:
                kwargs: Dict[str, Any] = {
                    "type": self.str_to_bool,
                    "default": opt.default,
                }
            else:
                kwargs: Dict[str, Any] = {"type": opt.type, "default": opt.default}
            if opt.nargs is not None:
                kwargs["nargs"] = opt.nargs
            if opt.choices is not None:
                kwargs["choices"] = opt.choices
            if opt.required:
                kwargs["required"] = opt.required
            if opt.metavar is not None:
                kwargs["metavar"] = opt.metavar
            if opt.action is not None:
                kwargs["action"] = opt.action
                if opt.action in ["store_true", "store_false"]:
                    kwargs.pop("type", None)
            parser.add_argument(*opt.flags, **kwargs)
        parser.add_argument("-l", "--log", type=str, default=None)
        parser.add_argument(
            "-v",
            "--verbose",
            type=str,
            default="WARNING",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)


class CLIFramework:
    def __init__(
        self,
        option_list: List[Union[Dict[str, Any], OptionConfig]],
        script_name: str,
        use_colors: bool = True,
        docs_root: Optional[Path] = None,
    ) -> None:
        self.script_name = script_name
        self.color_formatter = ColorFormatter(use_colors)
        self.doc_parser = DocumentationParser(script_name, docs_root)
        self.help_formatter = HelpFormatter()
        self.arg_parser = ArgumentParser(option_list)

    def generate_help_message(self) -> str:
        """Generate the help message from documentation."""
        try:
            sections = self.doc_parser.parse_documentation()
        except FileNotFoundError as e:
            log.error(f"Documentation file not found: {e}")
            return f"Help documentation not available for {self.script_name}"
        help_sections: List[str] = []
        section_order = [
            "Description",
            "Details",
            "Note",
            "Input Format",
            "Output Format",
            "Arguments",
            "Options",
            "Usage",
        ]
        for section_name in section_order:
            section = sections.get(section_name)
            if not section or section.is_empty():
                continue
            formatter_method = getattr(self.help_formatter, section.formatter, None)
            if not formatter_method:
                log.warning(f"No formatter found for section: {section_name}")
                continue
            formatted_content = formatter_method(section.content)
            if formatted_content:
                if section_name in [
                    "Arguments",
                    "Options",
                    "Usage",
                    "Note",
                    "Input Format",
                    "Output Format",
                ]:
                    help_sections.append(f"* {section_name}:\n{formatted_content}")
                else:
                    help_sections.append(f"* {section_name}: {formatted_content}")
        help_message = "\n\n".join(help_sections)
        return self.color_formatter.format_help_text(help_message)

    def run(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Run the CLI framework: parse args and handle help display."""
        check_args = args if args is not None else sys.argv
        if "-h" in check_args or "--help" in check_args:
            help_message = self.generate_help_message()
            print(help_message)
            sys.exit(0)
        parsed_args = self.arg_parser.parse_args(args)
        log.setup(level=parsed_args.verbose, file=parsed_args.log)
        return parsed_args
