from __future__ import annotations

import os
from typing import Final

import discord
from discord import app_commands

from agent import DEFAULT_MODEL, LocalAgentSession, OLLAMA_API_URL, memory_path_for_identity


BOT_TOKEN_ENV: Final[str] = "DISCORD_BOT_TOKEN"
GUILD_ID_ENV: Final[str] = "DISCORD_GUILD_ID"


class AgentDiscordBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.sessions: dict[str, LocalAgentSession] = {}

    def session_for_user(self, user_id: int) -> LocalAgentSession:
        key = str(user_id)
        if key not in self.sessions:
            self.sessions[key] = LocalAgentSession(memory_path_for_identity(key))
        return self.sessions[key]

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
    session = client.session_for_user(interaction.user.id)
    session.reset_history()
    await interaction.response.send_message("Your agent chat history has been reset.")


@client.tree.command(name="agent-forget-all", description="Delete your saved memory and reset chat history.")
async def forget_all_command(interaction: discord.Interaction) -> None:
    session = client.session_for_user(interaction.user.id)
    session.forget_all()
    await interaction.response.send_message("Your saved memory and chat history have been deleted.")


def main() -> int:
    token = os.environ.get(BOT_TOKEN_ENV)
    if not token:
        raise SystemExit(f"{BOT_TOKEN_ENV} is not set.")

    print(f"Discord bot starting with model: {DEFAULT_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_API_URL}")
    client.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
