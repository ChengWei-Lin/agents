from __future__ import annotations

import os
from typing import Final

import discord
from discord import app_commands

from agent import DEFAULT_MODEL, ENV_FILE_PATH, LocalAgentSession, OLLAMA_API_URL, memory_path_for_identity


BOT_TOKEN_ENV: Final[str] = "DISCORD_BOT_TOKEN"
GUILD_ID_ENV: Final[str] = "DISCORD_GUILD_ID"
ALLOWED_CHANNELS_ENV: Final[str] = "DISCORD_ALLOWED_CHANNEL_IDS"


def parse_allowed_channel_ids(raw_value: str | None) -> set[int]:
    if not raw_value:
        return set()
    channel_ids: set[int] = set()
    for part in raw_value.split(","):
        part = part.strip()
        if part.isdigit():
            channel_ids.add(int(part))
    return channel_ids


class AgentDiscordBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.sessions: dict[str, LocalAgentSession] = {}
        self.allowed_channel_ids = parse_allowed_channel_ids(os.environ.get(ALLOWED_CHANNELS_ENV))

    def session_for_user(self, user_id: int) -> LocalAgentSession:
        key = str(user_id)
        if key not in self.sessions:
            self.sessions[key] = LocalAgentSession(memory_path_for_identity(key))
        return self.sessions[key]

    def channel_allowed(self, interaction: discord.Interaction) -> bool:
        if not self.allowed_channel_ids:
            return True
        channel_id = interaction.channel_id
        return channel_id is not None and channel_id in self.allowed_channel_ids

    async def reject_if_disallowed(self, interaction: discord.Interaction) -> bool:
        if self.channel_allowed(interaction):
            return False
        allowed = ", ".join(str(channel_id) for channel_id in sorted(self.allowed_channel_ids))
        await interaction.response.send_message(
            f"This bot is restricted to configured channels only. Allowed channel IDs: {allowed}",
            ephemeral=True,
        )
        return True

    async def setup_hook(self) -> None:
        guild_id = os.environ.get(GUILD_ID_ENV)
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


client = AgentDiscordBot()


@client.tree.command(name="agent", description="Ask your local Ollama agent for help.")
@app_commands.describe(prompt="What you want the agent to help with")
async def agent_command(interaction: discord.Interaction, prompt: str) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    await interaction.response.defer(thinking=True)
    session = client.session_for_user(interaction.user.id)
    try:
        answer = session.run(prompt)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"Agent error: {exc}")
        return

    if len(answer) <= 1900:
        await interaction.followup.send(answer)
        return

    chunks = [answer[i : i + 1900] for i in range(0, len(answer), 1900)]
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.channel.send(chunk)


@client.tree.command(name="agent-reset", description="Reset your conversation history with the local agent.")
async def reset_command(interaction: discord.Interaction) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    session = client.session_for_user(interaction.user.id)
    session.reset_history()
    await interaction.response.send_message("Your agent chat history has been reset.", ephemeral=True)


@client.tree.command(name="agent-forget-all", description="Delete your saved memory and reset chat history.")
async def forget_all_command(interaction: discord.Interaction) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    session = client.session_for_user(interaction.user.id)
    session.forget_all()
    await interaction.response.send_message("Your saved memory and chat history have been deleted.", ephemeral=True)


@client.tree.command(name="agent-status", description="Show the current local bot configuration and memory status.")
async def status_command(interaction: discord.Interaction) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    session = client.session_for_user(interaction.user.id)
    memory_path = memory_path_for_identity(str(interaction.user.id))
    allowed_channels = (
        ", ".join(str(channel_id) for channel_id in sorted(client.allowed_channel_ids))
        if client.allowed_channel_ids
        else "all channels"
    )
    lines = [
        f"Model: `{DEFAULT_MODEL}`",
        f"Ollama endpoint: `{OLLAMA_API_URL}`",
        f"Guild sync ID: `{os.environ.get(GUILD_ID_ENV, 'not set')}`",
        f"Allowed channels: `{allowed_channels}`",
        f"Memory entries: `{session.memory_count()}`",
        f"Memory file: `{memory_path}`",
        f"Local config file: `{ENV_FILE_PATH}`",
    ]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


def main() -> int:
    token = os.environ.get(BOT_TOKEN_ENV)
    if not token:
        raise SystemExit(f"{BOT_TOKEN_ENV} is not set.")

    print(f"Discord bot starting with model: {DEFAULT_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_API_URL}")
    print(f"Allowed channel IDs: {sorted(client.allowed_channel_ids) or 'all'}")
    print(f"Local config file: {ENV_FILE_PATH}")
    client.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
