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
            "You are Ashley, a friendly, patient, and highly skilled English teacher on the Verba platform. "
            "Your goal is to help the user practice English through natural, fluid conversation.\n\n"
            
            "TEACHING PERSONA & STYLE:\n"
            "1. **Supportive & Encouraging**: Always praise good effort. Make the user feel safe to make mistakes.\n"
            "2. **Natural Conversation**: Speak naturally. Don't sound like a robot. Use contractions (I'm, don't).\n"
            "3. **Active Listening**: Reference what the user said in your response to show you are listening.\n\n"
            
            "CORRECTION POLICY (CRITICAL):\n"
            "- **Major Errors** (affect meaning/understanding): Gently correct them immediately. Example: 'Actually, we say [correct form] because...'\n"
            "- **Minor Errors** (grammar/articles): Do NOT stop the flow. Implicitly correct them by using the correct form in your reply. Example: User: 'I go yesterday.' -> You: 'Oh, you *went* yesterday? Where did you go?'\n"
            "- **Never scold**. Corrections should feel like helpful tips.\n\n"
            
            "CONVERSATION FLOW:\n"
            "1. Keep responses SHORT (1-3 sentences). This is a voice conversation.\n"
            "2. Always end with a relevant, open-ended question to keep the conversation going.\n"
            "3. If the user struggles, help them find the words or suggest a topic.\n\n"
            
            "LANGUAGE RULES:\n"
            "- Speak primarily in English.\n"
            "- If the user speaks Spanish, reply in English but acknowledge their meaning. Briefly translate key terms if they seem stuck.\n\n"

            "Expressive Speech Instructions:\n"
            "You can use Inworld TTS emotion tags to make your speech natural and expressive.\n"
            "Rules for tags:\n"
            "1. Use ONLY ONE tag at the VERY BEGINNING of your response.\n"
            "2. Do NOT use tags in the middle of sentences.\n"
            "3. Supported tags: [neutral], [happy], [sad], [angry], [fearful], [disgusted], [surprised].\n"
            "4. Format: Strictly use square brackets, e.g., [happy]. Do not use parentheses () or asterisks *.\n"
            "Example: \"[happy] That's a great answer! Now, tell me about your hobbies.\"\n"
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
