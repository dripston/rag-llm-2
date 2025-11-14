@echo off

echo Example curl commands for the Medical RAG Chatbot API
echo.

echo 1. Adding a medical document...
curl -X POST "http://localhost:8000/documents" ^
  -H "Content-Type: application/json" ^
  -d "{\"content\": \"The patient presented with fever, cough, and shortness of breath. Physical examination revealed crackles in the lower lung fields. Chest X-ray showed bilateral infiltrates. The diagnosis was community-acquired pneumonia. Treatment included antibiotics and supportive care.\", \"metadata\": {\"source\": \"medical_record\", \"patient_id\": \"P12345\", \"date\": \"2023-11-15\"}}"

echo.
echo.

echo 2. Querying the RAG system...
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What are the symptoms of pneumonia?\", \"top_k\": 3}"

echo.
echo.

echo 3. Adding another medical document...
curl -X POST "http://localhost:8000/documents" ^
  -H "Content-Type: application/json" ^
  -d "{\"content\": \"Hypertension management protocol includes lifestyle modifications such as dietary changes, regular exercise, and weight loss. Medications commonly prescribed include ACE inhibitors, beta-blockers, and diuretics. Regular monitoring of blood pressure is essential.\", \"metadata\": {\"source\": \"treatment_guidelines\", \"condition\": \"hypertension\", \"version\": \"2.1\"}}"

echo.
echo.

echo 4. Querying about hypertension...
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"How is hypertension managed?\", \"top_k\": 3}"

echo.
echo.

echo 5. Checking service health...
curl -X GET "http://localhost:8000/health"

echo.
echo Press any key to exit...
pause >nul