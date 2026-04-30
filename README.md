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
- `discord_bot.py` - Discord slash-command bot using the same local agent core
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

## Discord bot quick start

The project now includes a Discord bot entrypoint built on top of the same local Ollama agent.

Why slash commands first:

- easier to set up than raw message listeners
- no need to depend on message content intent for the first version
- safer and more predictable in shared servers

### What it does

- `/agent prompt:<your question>` asks the local model for help
- `/agent-reset` clears your current conversation history only
- `/agent-forget-all` deletes saved memory and resets chat history
- `/agent-status` shows the current model, endpoint, memory count, and channel restriction status
- each Discord user gets a separate local memory file in `memories/`

### Discord setup

1. Create a Discord application and bot in the Developer Portal.
2. Copy the bot token.
3. Invite the bot to your server with the `bot` and `applications.commands` scopes.
4. Install the Python dependency.
5. Run the bot locally on your machine.

Official references:

- [Discord Bots](https://discord.com/developers/docs/bots)
- [Discord quick start](https://docs.discord.com/developers/quick-start/getting-started)
- [discord.py quickstart](https://discordpy.readthedocs.io/en/v2.3.1/quickstart.html)

### Environment variables

- `DISCORD_BOT_TOKEN` - required
- `DISCORD_GUILD_ID` - optional but recommended during development for faster slash command sync in one server
- `DISCORD_ALLOWED_CHANNEL_IDS` - optional comma-separated channel IDs to restrict commands to specific channels
- `OLLAMA_MODEL` - optional, defaults to `qwen2.5:3b`
- `OLLAMA_API_URL` - optional, defaults to `http://localhost:11434/api/chat`
- `.env.local` - optional local config file that is auto-loaded by both `agent.py` and `discord_bot.py`

### Easy local config

To avoid re-exporting variables every time, copy `.env.local.example` to `.env.local` and fill in your values.

Git Bash:

```bash
cp ./.env.local.example ./.env.local
```

PowerShell:

```powershell
Copy-Item .\.env.local.example .\.env.local
```

You can then keep your Discord token, guild ID, default model, and optional allowed channel IDs in one private file.

Example:

```dotenv
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_API_URL=http://localhost:11434/api/chat
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=1490977470260707328
DISCORD_ALLOWED_CHANNEL_IDS=123456789012345678,234567890123456789
```

### Run the Discord bot

PowerShell:

```powershell
$env:Path="$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
python -m pip install -e .
$env:DISCORD_BOT_TOKEN="your_token_here"
$env:DISCORD_GUILD_ID="your_server_id"
python .\discord_bot.py
```

Git Bash:

```bash
export PATH="$LOCALAPPDATA/Programs/Ollama:$PATH"
/c/Python313/python.exe -m pip install -e .
/c/Python313/python.exe ./discord_bot.py
```

If you omit `DISCORD_GUILD_ID`, global slash command sync still works, but it can take longer to appear.
If you use `.env.local`, you usually do not need to export the Discord variables manually.

### Restricting the bot to one channel

If you only want the bot to work in one Discord channel, right-click that channel in Discord with Developer Mode enabled and copy its channel ID.

Then put it into `.env.local`:

```dotenv
DISCORD_ALLOWED_CHANNEL_IDS=123456789012345678
```

You can also allow multiple channels by separating IDs with commas.

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
