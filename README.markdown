# InsightFlow

InsightFlow is a FastAPI-based application with an AI-powered agentic workflow designed to analyze CSV files and images. It provides a web interface for uploading files, submitting queries, and receiving insights through data analysis and image text extraction. The project leverages LangChain and LangGraph for its agentic workflow, enabling intelligent processing of user queries.

## Features

- **CSV Analysis**: Upload CSV files and query them for summaries, averages, or other data insights.
- **Image Text Extraction**: Extract text from uploaded images (PNG, JPEG) using a vision model.
- **Mathematical Computations**: Perform simple calculations like division through natural language queries.
- **Session Management**: Maintains conversation history and file state within a session using in-memory storage.
- **File Cleanup**: Automatically deletes uploaded files older than 1 hour to manage disk space.
- **Web Interface**: A simple front-end for interacting with the API, uploading files, and viewing responses.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt` (see Installation section)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd insightflow
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` doesn’t exist yet, install the following packages:
   ```bash
   pip install fastapi uvicorn langchain-openai langgraph python-dotenv pandas
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your OpenAI API key (required for the vision model):
     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

## Project Structure

- `main.py`: FastAPI application setup, endpoints, and session management.
- `agent_workflow.py`: Agentic workflow logic using LangGraph, including tools for CSV analysis, image text extraction, and mathematical computations.
- `index.html`: Front-end web interface for interacting with the API.
- `uploads/`: Directory for storing uploaded files (created automatically).
- `agent.log`: Log file for debugging and monitoring (created automatically).

## Usage

1. **Run the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```
   - The `--reload` flag enables auto-reload for development.

2. **Access the Web Interface**:
   - Open your browser and navigate to `http://127.0.0.1:8000`.

3. **Interact with the Application**:
   - **Upload a File**:
     - Use the file input to upload a CSV or image (PNG, JPEG).
     - Supported file types: CSV, PNG, JPEG.
     - Maximum file size: 10MB.
   - **Submit a Query**:
     - Enter a query in the text input, e.g., "divide 6 by 2", "summarize data", or "what is in this image".
     - Click "Submit" to process the query.
   - **Clear CSV**:
     - Click "Clear CSV" to remove the currently loaded CSV file while retaining conversation history.
   - **Clear Session**:
     - Click "Clear Session" to reset the session, including conversation history and loaded files.

## Example Queries

- **Mathematical Computation**:
  - Query: "divide 6 by 2"
  - Expected Response: `3.0`

- **CSV Analysis**:
  - Upload a CSV file (e.g., `data.csv` with columns `name`, `age`, `salary`):
    ```
    name,age,salary
    Alice,25,50000
    Bob,30,60000
    Charlie,35,75000
    ```
  - Query: "summarize data"
  - Expected Response: A statistical summary of the CSV data.
  - Query: "average of column salary"
  - Expected Response: `Average of salary: 61666.666666666664`

- **Image Text Extraction**:
  - Upload an image containing text (e.g., a PNG with the text "Hello World").
  - Query: "what is in this image"
  - Expected Response: `Hello World` (or an error if the image is invalid).

## Limitations

- **Session Persistence**:
  - Session data is stored in-memory, so it will be lost if the server restarts.
  - For persistent storage, consider integrating a database like SQLite in the future.

- **Scalability**:
  - In-memory storage doesn’t scale across multiple server instances. For production, a shared storage solution is recommended.

- **File Cleanup**:
  - Files older than 1 hour are automatically deleted. Ensure queries are completed within this timeframe if they rely on uploaded files.

## Troubleshooting

- **API Key Error**:
  - If you see "Missing or invalid OPENAI_API_KEY", ensure your `.env` file contains a valid OpenAI API key.

- **No Response from Agent**:
  - Check `agent.log` for errors (e.g., "Graph execution failed").
  - Verify that the OpenAI API is accessible and your API key has sufficient credits.

- **File Upload Issues**:
  - Ensure the `uploads/` directory is writable.
  - Check file size (max 10MB) and type (CSV, PNG, JPEG).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.