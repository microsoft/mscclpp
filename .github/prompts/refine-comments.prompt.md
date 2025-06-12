---
description: 'Refine comments in the codebase.'
mode: agent
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'new', 'openSimpleBrowser', 'problems', 'runCommands', 'runNotebooks', 'runTasks', 'search', 'searchResults', 'testFailure', 'usages']
---
Your goal is refining comments in the codebase to improve clarity and understanding. For each code file, you will review the comments and suggest improvements. Especially pay attention to the following aspects:

- Clarity: Simplify complex comments and ensure they are easy to understand.
- Consistency: Maintain a consistent style and tone across comments.
- Relevance: Remove any comments that are no longer relevant or necessary.

Comply with the following regulations:

- Do not add more comments; focus on refining or deleting existing ones.
- Do not remove `TODO` keywords.
- Code snippets inside comments should be enclosed in backticks (`` ` ``) for inline code or triple backticks (```` ``` ````) for multi-line code blocks.
- Header files under `include/mscclpp/` directory should use Doxygen style comments. For these files, use `///` for both single-line and multi-line comments, and use `@p` for parameter descriptions.

Walk through the codebase, file by file, and refine the comments as needed. The target code file suffixes are: .cpp, .hpp, .cc, .h, .py, .cu

Now select the first code file in order to start refining comments. After refinement, review the changes and proceed to the next file. Do not stop until you are asked to do so.
