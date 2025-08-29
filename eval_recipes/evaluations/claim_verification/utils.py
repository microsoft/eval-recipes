# Copyright (c) Microsoft. All rights reserved.

from eval_recipes.evaluations.claim_verification.schemas import InputClaimVerificationEvaluator


class FormattedSource:
    """Handles line processing and formatting for a single source document."""

    def __init__(self, source_id: str, title: str, content: str, max_line_length: int) -> None:
        self.source_id = source_id
        self.title = title
        self.original_content = content
        self.lines = content.splitlines() if content else []

    def get_formatted_lines_with_numbers(self) -> str:
        """Return lines with line number prefixes, starting from 0, joined with newlines."""
        if not self.lines:
            return "The content is empty"

        # Calculate width for consistent formatting
        total_lines = len(self.lines)
        width = len(str(total_lines - 1)) if total_lines > 0 else 1

        formatted_lines = []
        for i, line in enumerate(self.lines):
            formatted_lines.append(f"{i:>{width}}→{line}")

        return "\n".join(formatted_lines)

    def get_text_by_range(self, start_range: int, end_range: int) -> str:
        """Extract original text from lines start_range (inclusive) to end_range (exclusive)."""
        if start_range >= end_range or start_range < 0 or start_range >= len(self.lines):
            return ""

        # Clamp end_range to valid range
        end_range = min(end_range, len(self.lines))

        # Return the selected lines joined with newlines to preserve original formatting
        selected_lines = self.lines[start_range:end_range]
        return "\n".join(selected_lines)


class FormattedContext:
    def __init__(self, input_data: InputClaimVerificationEvaluator, max_line_length: int) -> None:
        self.sources: list[FormattedSource] = []
        for ctx in input_data.source_context:
            formatted_source = FormattedSource(ctx.source_id, ctx.title, ctx.content, max_line_length)
            self.sources.append(formatted_source)

    def format_as_xml(self) -> str:
        """
        Generate XML representation with line numbers for all sources.

        Example output:
        <source id='1'>
        <title>User Message 0</title>
        <content>
        0→What is the capital of France?
        1→I need to know for my homework.
        </content>
        </source>
        """
        context_xml = ""
        for source in self.sources:
            context_xml += f"<source id='{source.source_id}'>\n"
            context_xml += f"<title>{source.title}</title>\n"
            context_xml += "<content>\n"
            context_xml += source.get_formatted_lines_with_numbers() + "\n"
            context_xml += "</content>\n"
            context_xml += "</source>\n\n"

        return context_xml.strip()

    def get_cited_text(self, source_id: str, start_range: int, end_range: int) -> str:
        """Extract cited text for the given source and line range."""
        for source in self.sources:
            if source.source_id == source_id:
                return source.get_text_by_range(start_range, end_range)
        return ""
