# Project README
@README.md


# Project Dependencies
The dependencies are defined in the pyproject.toml file:
@pyproject.toml

You must only use these dependencies in the code you write. If you need to add a new dependency, ask the user to set it up for you.


# Core Design Principles
### Ruthless Simplicity
- **KISS principle taken to heart**: Keep everything as simple as possible, but no simpler
- **Minimize abstractions**: Every layer of abstraction must justify its existence
- **Start minimal, grow as needed**: Begin with the simplest implementation that meets current needs
- **Avoid future-proofing**: Don't build for hypothetical future requirements
- **Question everything**: Regularly challenge complexity in the codebase

### Architectural Integrity with Minimal Implementation
- **Preserve key architectural patterns**: MCP for service communication, SSE for events, separate I/O channels, etc.
- **Simplify implementations**: Maintain pattern benefits with dramatically simpler code
- **Scrappy but structured**: Lightweight implementations of solid architectural foundations
- **End-to-end thinking**: Focus on complete flows rather than perfect components

### Library Usage Philosophy
- **Use libraries as intended**: Minimal wrappers around external libraries
- **Direct integration**: Avoid unnecessary adapter layers
- **Selective dependency**: Add dependencies only when they provide substantial value
- **Understand what you import**: No black-box dependencies

### Testing Strategy
- Emphasis on integration and end-to-end tests
- Manual testability as a design goal
- Focus on critical path testing initially
- Add unit tests for complex logic and edge cases

## Areas to Embrace Complexity
Some areas justify additional complexity:
1. **Security**: Never compromise on security fundamentals
2. **Data integrity**: Ensure data consistency and reliability
3. **Core user experience**: Make the primary user flows smooth and reliable
4. **Error visibility**: Make problems obvious and diagnosable

## Areas to Aggressively Simplify
Push for extreme simplicity in these areas:
1. **Internal abstractions**: Minimize layers between components
2. **Generic "future-proof" code**: Resist solving non-existent problems
3. **Edge case handling**: Handle the common cases well first
4. **Framework usage**: Use only what you need from frameworks
5. **State management**: Keep state simple and explicit

## Remember
- It's easier to add complexity later than to remove it
- Code you don't write has no bugs
- Favor clarity over cleverness
- The best code is often the simplest


# Development Guidelines
## Other Guidelines
- Do not use emojis unless asked.
- Do not include exceessive print and logging statements.
- You should only use the dependencies in the provided dependency files. If you need to add a new one, ask first.
- Do not automatically run scripts, tests, or move/rename/delete files. Ask the user to do these tasks.
- Read in the entirety of files to get the full context
- Include `# Copyright (c) Microsoft. All rights reserved` at the top of each Python file.

## Python Development Rules
- I am using Python version 3.12, uv as the package and project manager, and Ruff as a linter and code formatter.
- Follow the Google Python Style Guide.
- Instead of importing `Optional` from typing, using the `| `syntax.
- You **MUST** always add appropriate type hints such that the code would pass Pylance's type check.
- For type hints, use `list`, not `List`. For example, if the variable is `[{"name": "Jane", "age": 32}, {"name": "Amy", "age": 28}]` the type hint should be `list[dict[str. str | int]]`
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`.
- When using pathlib, **always** Use `.parents[i]` syntax to go up directories instead of using `.parent` multiple times.
- When writing multi-line strings, use `"""` instead of using string concatenation. Use `\` to break up long lines in appropriate places.
- When writing tests, use pytest and pytest-asyncio.
- Prefer to use pendulum instead of datetime
- Prefer to use loguru instead of logging
- Follow Ruff best practices such as:
  - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- Do not use relative imports.
- Use dotenv to load environment variables for local development. Assume we have a `.env` file

# Your workflow
- After making changes, `make check` will be called. If there are any linter or type errors, fix them.
