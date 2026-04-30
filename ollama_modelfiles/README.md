# Ollama Modelfiles

These files are starter templates for importing Taiwan-oriented GGUF models into Ollama.

Recommended first choice:

- `llama3-taiwan-8b.Modelfile`

Fallback if you want a lighter Traditional Chinese model:

- `breeze-7b.Modelfile`

How to use:

1. Download the matching GGUF file into this folder.
2. Rename the file so it matches the `FROM` line, or edit the `FROM` line to the exact file name.
3. Run `ollama create <your-model-name> -f <modelfile>`.

Examples:

```powershell
ollama create taiwan-tutor -f .\ollama_modelfiles\llama3-taiwan-8b.Modelfile
ollama create breeze-tutor -f .\ollama_modelfiles\breeze-7b.Modelfile
```

After that, point the app at the imported model with:

```dotenv
OLLAMA_MODEL=taiwan-tutor
```
