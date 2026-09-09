"""AI layer — GOJO's brains.

`base`  : the interface every provider must implement
`router`: picks the provider from config (routing rules grow here later)
`gemini`: Google Gemini provider (free tier) — the default brain
`mock`  : offline wiring test only, NOT a real AI
`prompts`: GOJO's personality
"""
