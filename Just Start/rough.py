import ollama
print(ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": "Test message"}]))
