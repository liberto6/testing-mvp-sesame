import os
from groq import AsyncGroq
from src.utils.config import Config

class LLMManager:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.client = None
        if self.api_key:
            try:
                self.client = AsyncGroq(api_key=self.api_key)
                print("[LLM] Groq Client initialized successfully.")
            except Exception as e:
                print(f"[LLM] Error initializing Groq client: {e}")
        
        self.system_prompt = (
            """
You are 'Sarah', a fun and engaging English tutor from London.

YOUR GOAL:

Balance strict grammar correction with a natural, friendly conversation. You must care about WHAT the student says, not just HOW they say it.

RESPONSE STRUCTURE (The "Sandwich" Method):

REACTION: React naturally to the user's story (e.g., "Oh, really?", "That sounds fun!").

CORRECTION (If needed): Explicitly compare the error. "Just a quick tip: You said 'X', but usually we say 'Y'".

CONVERSATION: Continue the topic and ask a relevant follow-up question.


EXAMPLES:

User: "Yesterday I go to the cinema."

You: "The cinema? Nice! Quick correction though: you said 'I go', but for the past we say 'I went'. What movie did you see?"

User: "I am boring." (Meaning 'bored')

You: "Oh no! Be careful: 'I am boring' means you are not interesting. You probably mean 'I am BORED'. Why? Do you have nothing to do today?"

RULES:

Never just correct. ALWAYS answer the content of the message too.

Keep it concise (max 3 sentences total).

Be encouraging.

"""
        )
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.max_history = 10  # Keep last 10 turns (5 user + 5 assistant)

    def _prune_history(self):
        """Keep only the system prompt and the last N turns."""
        if len(self.history) > self.max_history + 1:
            # Preserve system prompt (index 0)
            # Keep last max_history items
            self.history = [self.history[0]] + self.history[-self.max_history:]

    async def generate_response(self, text):
        """
        Generate a response for the given input text using Groq.
        """
        self.history.append({"role": "user", "content": text})
        self._prune_history()
        
        if self.client:
            try:
                # Using Llama 3.1 8b for speed/quality balance on Groq
                stream = await self.client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=self.history,
                    stream=True,
                    max_tokens=200,
                    temperature=0.7
                )
                
                full_response = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content
                
                self.history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                print(f"Error calling Groq API: {e}")
                error_msg = "I'm sorry, I'm having trouble connecting to the server right now."
                yield error_msg
                self.history.append({"role": "assistant", "content": error_msg})
        else:
            # Fallback
            response_text = f"Offline mode. You said: {text}"
            self.history.append({"role": "assistant", "content": response_text})
            yield response_text
