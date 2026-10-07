# LibraMind-AI

Software for the LibraMind AI LLM.

## LibraMind Chat

A browser chat app for **[LibraMind Mini](https://huggingface.co/alby13/LibraMindMini)**, a 565-million-parameter language model trained from scratch by alby13 on a single RTX 4090. LibraMind Mini is built for conversation and role-play, especially game NPCs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-dark.png">
  <img alt="LibraMind Chat: a conversation using the Friendly chat system prompt" src="docs/screenshot.png">
</picture>

## Features

- **Streaming replies**, about 90–130 tokens per second on an RTX 4090. Stop, regenerate, and edit-and-resend.
- **A system prompt you can change at any point.** It's sent with every message, so the next reply follows the new version. The conversation marks where it changed.
- **Ready-made prompts and characters:**
  - Friendly chat
  - Direct answers
  - Brom the blacksmith
  - Mira the innkeeper
  - Captain Vance (quest giver)
  - an NPC that replies in JSON

  You can save your own prompts too.
- **JSON mode.** Replies are forced to be valid JSON matching a schema you supply, with grammar-constrained decoding. That makes it easy to turn a character's reply into a game action such as `open_shop` or `give_item`.
- **Settings:** temperature, top-p, repetition penalty and longest reply. The defaults come from a blind test (see below).
- **A memory meter** for the 4,096-token context. When a long chat no longer fits, the oldest messages are dropped, but the system prompt is always kept.
- **Your conversation, settings and saved prompts stay in your browser**, so a reload keeps them.

## Requirements

- An **NVIDIA GPU** with about 2–3 GB of free VRAM. LibraMind's Gated DeltaNet layers run as Triton GPU kernels, so CPU-only and Apple Silicon machines aren't supported.
- Python 3.10 or newer (tested with 3.12)
- Windows or Linux

## Install

1. Get the code:

   ```bash
   git clone https://github.com/alby13/LibraMind-AI
   cd LibraMind-AI
   ```

2. Install PyTorch with CUDA by following [pytorch.org/get-started](https://pytorch.org/get-started/locally/). It was tested with PyTorch 2.14 and CUDA 13.0.
3. Install the other packages:

   ```bash
   pip install -r requirements.txt
   ```

On Windows, this also installs `triton-windows`, which provides the GPU kernels.

## Run

```bash
python chatui.py
```

The first run downloads the model from Hugging Face (about 1.1 GB). Your browser then opens at http://127.0.0.1:8052.

| Option | What it does |
|---|---|
| `--model PATH_OR_REPO` | A local copy of the model folder, or another Hugging Face repo (default: `alby13/LibraMindMini`) |
| `--port 8060` | Use a different port |
| `--host 0.0.0.0` | Let other devices on your network connect |
| `--no-browser` | Don't open a browser tab |

## Recommended settings

The defaults come from a blind test. LibraMind Mini answered 31 everyday conversations: small talk, advice, explanations, creative requests, practical tasks, opinions and multi-turn chats. It did so under 22 sampling settings and 8 system prompts. Grok 4.7 then scored every reply 1–10 without knowing which setting produced it.

- **Sampling: temperature 0.4, top-p 0.9, repetition penalty 1.1.**
  - Temperature 1.0 was clearly worse.
  - A repetition penalty of 1.2 hurt at temperature 0.6 and above.
  - The best settings between temperature 0.2 and 0.8 were close.
- **Use a system prompt.** Any reasonable one scored about a full point higher than none (5.3–5.6 vs 4.4 out of 10).
  - **Friendly chat** (the default) is the most consistent, and the best of the top prompts in multi-turn conversations.
  - **Direct answers** scored highest on explanations and creative requests, but is weaker in longer back-and-forth chats.

## Writing characters

Put the character card in the system prompt. Include who they are, where they are, how they speak, and **any facts they must get right**, such as prices, names and quest details. For example:

```
You are Mira, the cheerful innkeeper of the Gilded Goose inn in the village of Ashford. A room costs 5 silver a night and a bowl of stew costs 2 silver. You love local gossip. Stay in character, speak warmly, and keep replies to two or three sentences.
```

For game actions, switch **Reply format** to **JSON** and click **Use the NPC example**. That schema has the fields `line`, `emotion` (happy, angry, sad, neutral, surprised or afraid) and `action` (none, give_item, attack, open_shop, leave or follow_player). Describe the fields in the system prompt too, so the content makes sense as well as the format.

## Using it from a game or script

The page talks to a small local HTTP API, and you can call it from your own code.

`POST /api/chat` with a JSON body:

```json
{
  "system": "You are Brom, a gruff dwarven blacksmith...",
  "messages": [{"role": "user", "content": "Can you fix my sword?"}],
  "temperature": 0.4, "top_p": 0.9, "rep_penalty": 1.1, "max_new": 256,
  "schema": {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"]}
}
```

`schema` is optional; add it to force JSON output. The response streams as one JSON object per line:
- first `{"start": true, "prompt_tokens": ..., "dropped": ...}`
- then `{"delta": "..."}` pieces of text
- finally `{"done": true, "text": "<the full reply>", "tokens": ..., "seconds": ...}`

Close the connection to stop a reply early.

`GET /api/status` returns the model's name, size and context length.

The server answers one request at a time and is meant for your own machine. Don't expose it to the internet.

## Limitations

LibraMind Mini is a small model trained on a hobby budget:
- **Facts:** it often states wrong or invented facts confidently.
- **Math:** it fails at arithmetic beyond single digits.
- **Code:** it rarely writes correct code.
- **Memory:** in long conversations it loses track of details mentioned more than a few hundred words earlier.

It's good company for chatting and characters, but check anything that matters, and compute numbers such as prices and totals in your own code. There was no dedicated safety training, so filter its output before showing it to the public. See the [model card](https://huggingface.co/alby13/LibraMindMini) for full evaluations.

## Files

| File | What it is |
|---|---|
| `chatui.py` | The local server: loads the model, streams replies, applies JSON schemas |
| `chatui.html` | The chat page |
| `model.py`, `inference.py`, `chat_format.py` | LibraMind Mini's architecture, fast generation (cached and CUDA-graph decoding, grammar-forced JSON), and chat template |

## License

This software may be used and modified only to run the LibraMind Mini AI model. See [LICENSE](LICENSE).

The model weights are published separately on [Hugging Face](https://huggingface.co/alby13/LibraMindMini) under their own license. See the model card.
