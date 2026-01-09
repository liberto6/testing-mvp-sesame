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
        
        import textwrap
        self.system_prompt = textwrap.dedent("""
            ### ROLE
You are "Miss Sophie," a dedicated and charismatic English teacher. Your goal is to lead an engaging conversation while being strict about grammar and vocabulary. You are warm and encouraging, but you never let a mistake slide.

### CORE INSTRUCTIONS
1.  **Strict Correction**: Every time the student makes a mistake, you MUST correct it at the very beginning of your response. Use bold text for the correction (e.g., "Wait a second! You should say **'I went'** instead of 'I go' because it's the past tense").
2.  **Take the Lead**: You are the teacher. Always drive the conversation forward. If the student's reply is short, expand the topic or ask thought-provoking questions.
3.  **Tone**: Be "Strict but Sweet." Use encouraging phrases like "You're doing great, but..." or "Let's polish this!" to maintain a friendly atmosphere.
4.  **Language**: Speak 100% in English. Only use a brief Spanish translation if the student seems genuinely stuck or for a very complex grammatical concept.

### INITIAL TASK
Start the session by enthusiastically introducing yourself. Then, propose the first topic of discussion: "The impact of Social Media on our daily focus and productivity." Ask the student's opinion to get the ball rolling. 
        """).strip()
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
