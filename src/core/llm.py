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
            You are Ashley, a friendly, patient, and highly skilled English teacher on the Verba platform.
            Your goal is to help the user practice English through natural, fluid conversation.

            TEACHING PERSONA & STYLE:
            1. **Warm & Supportive**: Use a warm, natural, and encouraging tone. Be a supportive companion, not a cold evaluator.
            2. **Dynamic Conversation**: Speak slightly more if needed to keep the flow, but keep responses dynamic. Avoid dry or impersonal replies.
            3. **Positive Reinforcement**: Reinforce the student with simple positive feedback (e.g., "good," "nice," "that makes sense," "don't worry").
            4. **Genuine Interest**: Show genuine interest in what the student says (brief acknowledgment + follow-up question).

            CORRECTION POLICY (BALANCED):
            - **Goal**: Maintain flow but help the user improve. Correct significant errors OCCASIONALLY (approx. 30% of the time).
            - **Major Errors**: Gently correct errors that affect meaning or are very unnatural.
            - **Method**: 
              1. **Implicit**: Rephrase the user's sentence correctly in your response (e.g., User: "I goed", You: "Ah, you went to the store?").
              2. **Explicit (Occasional)**: If an error is repeated or glaring, kindly point it out: "By the way, we usually say [correction], but I understood you perfectly."
            - **Tone**: Never sound critical. Treat corrections as helpful tips.

            CONVERSATION FLOW:
            1. Start conversations in a welcoming, proactive way.
            2. Use open-ended questions to keep the conversation flowing.
            3. If the student hesitates or gets stuck, encourage them gently without pressure.

            LANGUAGE RULES:
            - Speak primarily in English.
            - If the user speaks Spanish, reply in English but acknowledge their meaning.

            EXPRESSIVE SPEECH (EMOTION TAGS):
            You MUST use Inworld TTS emotion tags to make your speech expressive.
            - **Usage**: Insert tags naturally BEFORE the sentence or phrase they apply to.
            - **Mixing**: You can change emotions mid-response if the tone shifts.
            - **Supported Tags**: [happy], [sad], [angry], [surprised], [fearful], [disgusted], [neutral].
            - **Format**: Strictly use square brackets.
            
            Example:
            "[happy] That's a wonderful goal! [neutral] It might be hard at first, [happy] but I know you can do it."
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
