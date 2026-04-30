# Starter Agent with Codex

This repo is a small local agent you can study and extend yourself.

It is intentionally much smaller than Hermes Agent, but it borrows the same core idea:

- the model can decide when to use tools
- the agent keeps lightweight memory across turns
- the loop is explicit, so you can understand how it works

## What this project includes

- a terminal chat loop
- Ollama local model integration
- function calling for local tools
- persistent JSON memory
- a simple "save what matters" pattern you can inspect and change

## Files

- `pyproject.toml` - project metadata
- `agent.py` - runnable CLI agent
- `tests/test_memory_store.py` - small tests for the memory layer

## Quick start

1. Install Ollama.
2. Pull a local model.
3. Run the agent.

```powershell
$env:Path="$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
ollama pull qwen2.5:3b
python .\agent.py
```

Optional model override:

```powershell
$env:OLLAMA_MODEL="qwen2.5:7b"
python .\agent.py
```

Type `exit` or `quit` to stop.

Optional custom Ollama endpoint:

```powershell
$env:OLLAMA_API_URL="http://localhost:11434/api/chat"
python .\agent.py
```

## What the agent can do

Today it has four local tools:

- `get_time`
- `save_memory`
- `list_memories`
- `search_memories`

The model can call these tools when useful. For example, if you tell it your preferred stack or ask it to remember a project decision, it can store that in `memory.json`.

## How this compares to Hermes Agent

Hermes Agent is much more ambitious. Based on the current README and docs, Hermes includes:

- persistent memory and user modeling
- auto-created and self-improving skills
- many built-in tools
- messaging platform gateways
- automations and cron
- MCP integration
- subagents and parallel work
- remote and sandboxed execution backends

Sources:

- [Hermes Agent README](https://github.com/nousresearch/hermes-agent/blob/main/README.md)
- [Hermes Agent Docs](https://hermes-agent.nousresearch.com/docs/)

This starter project does **not** try to compete with that. It gives you the smallest understandable slice:

- one agent
- one tool loop
- one memory file
- one place to customize behavior

That makes it a good learning project.

## How the loop works

At a high level:

1. You send a message.
2. The app sends your message plus tool schemas to Ollama's local chat API.
3. If the model requests a function call, the app runs it locally.
4. The app sends the tool output back to Ollama.
5. The model returns the final answer.

This pattern follows Ollama's current chat API and tool-calling guidance for local agent-style apps.

Official references:

- [Ollama Quickstart](https://docs.ollama.com/quickstart)
- [Ollama Chat API](https://docs.ollama.com/api/chat)
- [Ollama Tool Calling](https://docs.ollama.com/capabilities/tool-calling)

## Good next upgrades

If you want, we can extend this into any of these next:

- file search over a project folder
- a task list and daily planning tool
- web research tools
- a small local knowledge base with embeddings
- a Discord or Telegram interface
- a multi-agent coordinator
- a Hermes-style skill loader from markdown files

## Offline behavior

After you install Ollama and download a model, the agent can run offline.

What still needs internet:

- downloading Ollama the first time
- downloading models the first time
- any future online tools you choose to add, such as Discord or web search

So the current terminal agent can be fully local and offline once setup is done.

## Recommended model for this laptop

This project is currently tuned for:

- `qwen2.5:3b` as the safest default

Later, you can try:

- `qwen2.5:7b` for stronger quality if performance still feels good

If you want the model files on another drive, Ollama supports setting `OLLAMA_MODELS` to a custom folder such as `D:\OllamaModels`.

## Windows note for Hermes

Hermes' current README says native Windows is not supported and recommends WSL2 for running Hermes itself.

This starter project is plain Python and should run directly on Windows.
