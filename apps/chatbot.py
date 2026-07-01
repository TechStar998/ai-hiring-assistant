import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from apps.prompts import SYSTEM_PROMPT, build_user_prompt
from apps.retriever import SHLRetriever

load_dotenv()


class SHLChatbot:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

        self.retriever = SHLRetriever()

    def _format_conversation(self, messages):
        conversation = []

        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            conversation.append(f"{role}: {content}")

        return "\n".join(conversation)

    def _format_retrieved_results(self, results):
        formatted = []

        for item in results:
            formatted.append(
                f"""
Name: {item.get("name", "")}
URL: {item.get("url", "")}
Test Type: {item.get("test_type", "")}
Description: {item.get("description", "")}
"""
            )

        return "\n-----------------------------\n".join(formatted)

    def _call_gemini(self, conversation, retrieved_catalog):
        prompt = build_user_prompt(
            conversation=conversation,
            retrieved_catalog=retrieved_catalog,
        )

        response = self.model.generate_content(
            [
                SYSTEM_PROMPT,
                prompt,
            ]
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        return text

    def chat(self, messages):
        conversation = self._format_conversation(messages)

        latest_user_message = ""

        for message in reversed(messages):
            if message["role"] == "user":
                latest_user_message = message["content"]
                break

        retrieved = self.retriever.search(
            latest_user_message,
            top_k=10,
        )

        retrieved_catalog = self._format_retrieved_results(retrieved)

        raw_response = self._call_gemini(
            conversation,
            retrieved_catalog,
        )

        try:
            result = json.loads(raw_response)

        except json.JSONDecodeError:
            result = {
                "reply": "Sorry, I couldn't generate a valid response.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        result.setdefault("reply", "")

        result.setdefault("recommendations", [])

        result.setdefault("end_of_conversation", False)

        return result