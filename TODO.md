# TODO

## Chinese Tutor

- Build a dedicated structured Chinese lesson path for `language_tutor` instead of relying on normal free-form chat output.
- Detect Chinese-teaching intents such as:
  - "teach me Chinese"
  - greeting / vocabulary / phrase requests
  - beginner Chinese lesson requests
- For those intents, bypass the normal answer flow and generate structured lesson entries first.

## Structured Output

- Render Chinese lesson entries in a fixed template:
  - `phrase (Zhuyin) (Pinyin)`
  - `Meaning:`
  - `- character or word gloss lines in English`
  - `- whole phrase meaning and usage line in English`
- Ensure beginner explanations default to English.
- Keep Traditional Chinese as the default script.
- Keep both Zhuyin and Pinyin for German learners unless the user requests only one system.

## Deterministic Processing

- Keep deterministic Zhuyin generation in code.
- Add deterministic Pinyin generation in the final rendered output.
- Add stronger normalization to block Simplified Chinese from leaking into beginner Chinese lessons.
- Consider adding OpenCC for Traditional Chinese normalization.

## Model Reliability

- Reduce dependence on model-generated formatting.
- Use the model only for compact semantic extraction when necessary:
  - phrase meaning
  - gloss suggestions
  - usage note
- Keep final rendering fully code-driven whenever possible.

## Testing

- Add tests for structured Chinese lesson rendering.
- Add tests for intent detection that routes into the dedicated Chinese lesson path.
- Add tests for Traditional Chinese normalization.
- Add tests for mixed Zhuyin + Pinyin beginner output.

## Nice-to-Have

- Add a beginner / intermediate / advanced lesson mode for Chinese specifically.
- Add a setting for explanation language, e.g. English or German.
- Add a small deterministic glossary for common greetings and survival phrases.
