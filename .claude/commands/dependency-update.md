I need you to update the dependencies in pyproject.toml.
The process to go through is:
1. Notice how we do versioning. We make sure that we do not auto upgrade to the major version. For example, "openai[aiohttp]>=2.9,<3.0", means we will never upgrade to v3.x. You will just be updating minor versions. Note that for dev dependencies we don't need to pin to be less than the major version. Additionally, do not touch uv_build or anything outside `dependencies` or `[dependency-groups]`
2. For each dependency, go to its pypi release history site. For example, for the openai package that is: https://pypi.org/project/openai/#history Get the latest release version.
3. Now bump the dependency in pyproject.toml. For example, if the current version in the pyproject.toml is `>=1.05,<2.0`, but on Pypi the latest version is 1.11, change the dependency to `>=1.11,<2.0`
4. If you notice a major version upgrade (ex v2 to v3), let the user know of each of those cases, but do not make the change yourself.
5. Run `uv sync --all-extras --all-groups` to update the lock file.
