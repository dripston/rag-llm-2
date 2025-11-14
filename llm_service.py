import os
import requests
import json
from typing import List, Dict, Any

class SambaNovaLLM:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL")
        self.base_url = "https://api.sambanova.ai/v1"
        
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response using SambaNova's LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating response: {response.text}")
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_response_with_context(self, context: str, query: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response using context and query
        
        Args:
            context: Retrieved context from RAG
            query: User query
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Use the provided context to answer the user's question accurately."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            }
        ]
        
        return self.generate_response(messages, temperature, max_tokens)