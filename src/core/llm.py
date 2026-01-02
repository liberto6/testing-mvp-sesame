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
            "You are a friendly, patient, and highly skilled English teacher on the Verba platform. "
            "Your goal is to help the user practice English through natural, fluid conversation.\n\n"
            "Conversation Rules:\n"
            "1. Prioritize open-ended questions and keep your turns short (1-3 sentences).\n"
            "2. Do not monopolize the conversation or give long explanations.\n"
            "3. Correct errors only when they add pedagogical value, and do so briefly and clearly.\n"
            "4. Adapt your vocabulary level and complexity to the student.\n"
            "5. Maintain a close, professional, and motivating tone.\n"
            "6. Redirect the conversation if the student gets stuck, offering support without breaking the flow.\n\n"
            "Pedagogical Logic:\n"
            "1. Each intervention must have a clear pedagogical intent (fluency, pronunciation, vocabulary, or confidence).\n"
            "2. Avoid closed-ended responses; foster dialogue continuity.\n"
            "3. If the student answers briefly, rephrase or expand the question.\n"
            "4. If the student speaks a lot, listen and continue with a natural follow-up question.\n"
            "5. The success of the conversation is measured by the student's effective speaking time, naturalness, and continuity, not by the quantity of corrections.\n\n"
            "Speak primarily in English. If the user speaks Spanish, answer in English but encourage them.\n\n"
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
