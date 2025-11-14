import logging
import os
import requests
import json
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class SambaNovaLLM:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL")
        self.base_url = "https://api.sambanova.ai/v1"
        
        logger.info("Initializing SambaNova LLM service...")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Base URL: {self.base_url}")
        
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
        logger.info("Generating response with LLM")
        logger.debug(f"Number of messages: {len(messages)}")
        logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")
        
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
        
        logger.debug("Calling SambaNova chat completions API...")
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error generating response: {response.text}")
            logger.error(f"Status code: {response.status_code}")
            raise Exception(f"Error generating response: {response.text}")
            
        data = response.json()
        result = data["choices"][0]["message"]["content"]
        logger.info(f"Response generated successfully. Response length: {len(result)}")
        return result
    
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
        logger.info("Generating response with context")
        logger.debug(f"Context length: {len(context)}, Query length: {len(query)}")
        logger.info(f"Context passed to LLM: {context[:200]}...")  # Log first 200 chars of context
        
        # If no context returned, avoid empty answer
        if not context or context.strip() == "":
            context = "No relevant medical history found in the database."
        
        # Create a cleaner prompt format
        system_message = "You are an advanced medical RAG assistant. You must strictly use the provided context for every answer. If the answer is not found in the context, say 'I could not find this information in the patient's medical records.'"
        
        user_message = f"""Context:
{context}

Question:
{query}"""
        
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        logger.info("Embedding model used for QUERY: E5-Mistral-7B-Instruct")  # Added required logging
        return self.generate_response(messages, temperature, max_tokens)