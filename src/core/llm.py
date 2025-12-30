import os
from groq import Groq
from src.utils.config import Config

class LLMManager:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                print("[LLM] Groq Client initialized successfully.")
            except Exception as e:
                print(f"[LLM] Error initializing Groq client: {e}")
        
        self.system_prompt = (
            "You are a friendly, patient, and highly skilled English teacher. "
            "Your goal is to help the user practice English through natural, fluid conversation. "
            "1. Speak primarily in English. "
            "2. Keep your responses concise (1-3 sentences) to maintain a fast conversational pace. "
            "3. If the user makes a mistake, gently correct them or rephrase it correctly in your reply, but don't be pedantic. "
            "4. If the user speaks Spanish, answer their question but encourage them to try saying it in English. "
            "5. Be engaging and ask follow-up questions to keep the dialogue going."
        )
        self.history = [{"role": "system", "content": self.system_prompt}]

    async def generate_response(self, text):
        """
        Generate a response for the given input text using Groq.
        """
        self.history.append({"role": "user", "content": text})
        
        if self.client:
            try:
                # Using Llama 3.1 8b for speed/quality balance on Groq
                stream = await self.client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=self.history,
                    stream=True,
                    max_tokens=150,
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
