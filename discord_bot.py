from __future__ import annotations

import os
from typing import Final

import discord
from discord import app_commands

from agent import (
    DEFAULT_MODEL,
    DEFAULT_PERSONA,
    DEFAULT_TUTOR_MODE,
    ENV_FILE_PATH,
    LocalAgentSession,
    OLLAMA_API_URL,
    PERSONAS,
    TUTOR_MODES,
    memory_path_for_identity,
    normalize_persona,
    normalize_tutor_mode,
)


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
        self.sessions: dict[tuple[str, str, str], LocalAgentSession] = {}
        self.default_personas: dict[str, str] = {}
        self.default_tutor_modes: dict[str, str] = {}
        self.allowed_channel_ids = parse_allowed_channel_ids(os.environ.get(ALLOWED_CHANNELS_ENV))

    def persona_for_user(self, user_id: int) -> str:
        key = str(user_id)
        return self.default_personas.get(key, DEFAULT_PERSONA)

    def set_persona_for_user(self, user_id: int, persona: str) -> str:
        key = str(user_id)
        chosen = normalize_persona(persona)
        self.default_personas[key] = chosen
        return chosen

    def tutor_mode_for_user(self, user_id: int) -> str:
        key = str(user_id)
        return self.default_tutor_modes.get(key, DEFAULT_TUTOR_MODE)

    def set_tutor_mode_for_user(self, user_id: int, tutor_mode: str) -> str:
        key = str(user_id)
        chosen = normalize_tutor_mode(tutor_mode)
        self.default_tutor_modes[key] = chosen
        return chosen

    def session_for_user(
        self,
        user_id: int,
        persona: str | None = None,
        tutor_mode: str | None = None,
    ) -> LocalAgentSession:
        user_key = str(user_id)
        persona_key = normalize_persona(persona or self.persona_for_user(user_id))
        tutor_mode_key = normalize_tutor_mode(tutor_mode or self.tutor_mode_for_user(user_id))
        session_key = (user_key, persona_key, tutor_mode_key)
        if session_key not in self.sessions:
            self.sessions[session_key] = LocalAgentSession(
                memory_path_for_identity(user_key),
                persona=persona_key,
                tutor_mode=tutor_mode_key,
            )
        return self.sessions[session_key]

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
PERSONA_CHOICES = [
    app_commands.Choice(name=persona.label, value=persona.key)
    for persona in PERSONAS.values()
]
TUTOR_MODE_CHOICES = [
    app_commands.Choice(name=mode.label, value=mode.key)
    for mode in TUTOR_MODES.values()
]


@client.tree.command(name="agent", description="Ask your local Ollama agent for help.")
@app_commands.describe(prompt="What you want the agent to help with")
@app_commands.describe(persona="Optional persona override for this message")
@app_commands.describe(tutor_mode="Optional lesson mode for language_tutor")
@app_commands.choices(persona=PERSONA_CHOICES)
@app_commands.choices(tutor_mode=TUTOR_MODE_CHOICES)
async def agent_command(
    interaction: discord.Interaction,
    prompt: str,
    persona: app_commands.Choice[str] | None = None,
    tutor_mode: app_commands.Choice[str] | None = None,
) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    await interaction.response.defer(thinking=True)
    session = client.session_for_user(
        interaction.user.id,
        persona.value if persona else None,
        tutor_mode.value if tutor_mode else None,
    )
    try:
        answer = session.run(prompt)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"Agent error: {exc}")
        return

    header = ""
    if persona is not None or (tutor_mode is not None and session.persona_key == "language_tutor"):
        header = f"[{session.persona_label()}"
        if session.persona_key == "language_tutor":
            header += f" | {session.tutor_mode_label()}"
        header += "]\n"
        answer = f"{header}{answer}"

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
    await interaction.response.send_message(
        f"Your {session.persona_label()} chat history has been reset.",
        ephemeral=True,
    )


@client.tree.command(name="agent-forget-all", description="Delete your saved memory and reset chat history.")
async def forget_all_command(interaction: discord.Interaction) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    user_key = str(interaction.user.id)
    for persona_key in PERSONAS:
        for tutor_mode_key in TUTOR_MODES:
            client.session_for_user(interaction.user.id, persona_key, tutor_mode_key).forget_all()
    client.sessions = {
        key: value for key, value in client.sessions.items() if key[0] != user_key
    }
    await interaction.response.send_message("Your saved memory and chat history have been deleted.", ephemeral=True)


@client.tree.command(name="agent-persona", description="Set your default persona for future agent chats.")
@app_commands.describe(persona="Which persona should be your default")
@app_commands.choices(persona=PERSONA_CHOICES)
async def persona_command(interaction: discord.Interaction, persona: app_commands.Choice[str]) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    chosen = client.set_persona_for_user(interaction.user.id, persona.value)
    session = client.session_for_user(interaction.user.id, chosen)
    await interaction.response.send_message(
        f"Default persona set to {session.persona_label()} (`{chosen}`). Shared memory stays the same across personas.",
        ephemeral=True,
    )


@client.tree.command(name="agent-mode", description="Set your default tutor mode for language learning chats.")
@app_commands.describe(tutor_mode="Which language tutor mode should be your default")
@app_commands.choices(tutor_mode=TUTOR_MODE_CHOICES)
async def tutor_mode_command(
    interaction: discord.Interaction,
    tutor_mode: app_commands.Choice[str],
) -> None:
    if await client.reject_if_disallowed(interaction):
        return
    chosen = client.set_tutor_mode_for_user(interaction.user.id, tutor_mode.value)
    session = client.session_for_user(interaction.user.id, "language_tutor", chosen)
    await interaction.response.send_message(
        f"Default language tutor mode set to {session.tutor_mode_label()} (`{chosen}`).",
        ephemeral=True,
    )


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
        f"Default persona: `{session.persona_key}` ({session.persona_label()})",
        f"Available personas: `{', '.join(sorted(PERSONAS))}`",
        f"Default tutor mode: `{client.tutor_mode_for_user(interaction.user.id)}`",
        f"Available tutor modes: `{', '.join(sorted(TUTOR_MODES))}`",
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
