import json
from typing import Dict, Any

class WebhookProcessor:
    """
    Process webhooks from PostgreSQL database and update RAG system
    """
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
    
    def process_webhook(self, data: Dict[str, Any]) -> bool:
        """
        Process webhook data and update RAG system accordingly
        
        Args:
            data: Webhook data from PostgreSQL
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            event_type = data.get("event", "")
            table_name = data.get("table", "")
            record_data = data.get("data", {})
            
            if event_type == "INSERT":
                return self._handle_insert(table_name, record_data)
            elif event_type == "UPDATE":
                return self._handle_update(table_name, record_data)
            elif event_type == "DELETE":
                return self._handle_delete(table_name, record_data)
            else:
                print(f"Unknown event type: {event_type}")
                return False
                
        except Exception as e:
            print(f"Error processing webhook: {e}")
            return False
    
    def _handle_insert(self, table_name: str, record_data: Dict[str, Any]) -> bool:
        """
        Handle INSERT events
        """
        try:
            # Convert record data to a document for RAG
            content = self._format_record_as_document(table_name, record_data)
            metadata = {
                "table": table_name,
                "operation": "INSERT",
                "record_id": record_data.get("id", ""),
                "timestamp": record_data.get("created_at", "")
            }
            
            return self.rag_service.add_document(content, metadata)
        except Exception as e:
            print(f"Error handling INSERT event: {e}")
            return False
    
    def _handle_update(self, table_name: str, record_data: Dict[str, Any]) -> bool:
        """
        Handle UPDATE events
        """
        try:
            # For updates, we might want to update the existing document
            # This would require tracking document IDs
            content = self._format_record_as_document(table_name, record_data)
            metadata = {
                "table": table_name,
                "operation": "UPDATE",
                "record_id": record_data.get("id", ""),
                "timestamp": record_data.get("updated_at", "")
            }
            
            # In a real implementation, you would track the document ID
            # and update the existing document
            # For now, we'll just add as a new document
            return self.rag_service.add_document(content, metadata)
        except Exception as e:
            print(f"Error handling UPDATE event: {e}")
            return False
    
    def _handle_delete(self, table_name: str, record_data: Dict[str, Any]) -> bool:
        """
        Handle DELETE events
        """
        try:
            # For deletes, we would remove the document from the RAG system
            # This would require tracking document IDs
            print(f"Delete event for table {table_name}: {record_data}")
            # In a real implementation, you would delete the corresponding document
            return True
        except Exception as e:
            print(f"Error handling DELETE event: {e}")
            return False
    
    def _format_record_as_document(self, table_name: str, record_data: Dict[str, Any]) -> str:
        """
        Format database record as a document for the RAG system
        """
        # This is a simple implementation - you would customize this based on your data structure
        document_parts = [f"Table: {table_name}"]
        
        for key, value in record_data.items():
            if key not in ["id", "created_at", "updated_at"]:
                document_parts.append(f"{key}: {value}")
        
        return "\n".join(document_parts)