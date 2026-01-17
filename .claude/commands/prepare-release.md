I am about to release a new version of this package. Please take the following steps to make sure it is successful:

Understand changes
1. Look at my currently staged changes to identify what changes were made.

Checks
1. Make sure the pyproject.toml was updated with a new version number.
2. Ensure there are no spelling and grammar mistakes.
3. Run all formatting, linting, and type checking: `make check`
4. Run `uv build` to make sure the package builds correctly.
5. If any of these checks fail, please stop and inform me about the issues so we can fix them before proceeding.

Draft release notes
1. Look at the previous release logs at https://github.com/microsoft/eval-recipes/releases your draft release MUST follow the same style and structure.
2. Create a draft release description based on the recent code changes and place it in `media/draft_release_{version}.md`
3. At the end of the release notes be sure to include:
```
**Full Changelog**:  https://github.com/microsoft/eval-recipes/compare/v0.x1.y1...v0.x2.y2
```