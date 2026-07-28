## General

- Always review the repository @README.md to orient yourself with this project.
- Keep code DRY, elegant, and simple.
- Do not worry about code formatting or linting, since that will be handled automatically by `prek` pre-commit hooks.

## Version Control

Never stage or commit files unless explicitly told to do so.
The user will handle all Git write operations.

When asked to draft a commit message, see the last several commit messages and current staged and unstaged diffs.
Scope the message to just work that has not yet been committed, which is often less than scope of the coding session.

Commit messages should use full https issue or PR URLs as described below.

## GitHub

Never post comments to GitHub unless the user explicitly asks you to publish.
Usually, you should instead write an `.md` file to the `local` directory that the user can copy-paste into GitHub after reviewing.

For draft GitHub comments, always use bare full (versioned) URLs like:

- `https://github.com/dhimmel/openskistats/issues/1` not `#1`
- `https://github.com/dhimmel/openskistats/blob/8955765a52fc2bf11a568c1a1248807d0df1cc04/README.md#L9-L10` not `README.md`

This ensures portability and takes advantage of GitHub flavored markdown and its rendering features such as autolinking and code snippets.

## Execution

To ensure access to the environment, prefix commands with `pixi run`.

Unless explicitly instructed to, never run commands that install any software or make changes outside of this repository (temporary directories are okay).
Flag any such commands that are required and ask the user to run them separately.

## Testing

The test suite is executed with [pytest](https://docs.pytest.org/en/stable/).
We will always be using the latest pytest version or can upgrade to it and want to take advantage of the newest features and syntactic sugar.

Use `@pytest.mark.parametrize` when relevant with `pytest.param` to set self-documenting `id` values that explain the purpose of the test parameters.

Some pytest invocations can hang in the Codex tool environment even when the test itself is fine.
If pytest stops making progress after `functions.exec_command`, do not assume the test needs to be rewritten.
Instead give the user a command to run the hung test when you are done.
CI is the final arbiter of whether the test passes.

## Python

- Write docstrings and comments using Markdown conventions rather than reStructuredText.
  Use single backticks for inline code, not reStructuredText double-backtick literals.
  Prefer raw docstrings for backslash-heavy examples so the source remains readable without doubled backslashes.
  Use semantic newlines, one sentence/phrase per line, when deciding where to wrap comments and docstrings.
- Use `pathlib.Path` for local file paths with `.joinpath` rather than `__truediv__` slashes for path construction.
- Prefer a named structured type (a frozen dataclass or NamedTuple) over a bare tuple when a function returns multiple heterogeneous values.
  Callers should reach fields by attribute, never by positional unpacking of an undocumented tuple.
- When a function's behavior is primarily a dispatch on, or a derivation from, a single StrEnum, dataclass, or Pydantic model, define it as a method on that class rather than a module-level function that takes the instance as an argument.
- Put imports at module top level. Only move an import inside a function to break a circular with confirmed cycles or to defer a genuinely expensive or optional dependency.
