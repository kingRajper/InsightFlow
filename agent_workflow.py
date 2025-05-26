from typing import TypedDict, List, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import logging
import base64
import mimetypes
import os
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize vision model
try:
    vision_model = ChatOpenAI(model='gpt-4o')
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
    raise Exception("Missing or invalid OPENAI_API_KEY. Please check your .env file.")

# Step 1: Define Agent's State
class AgentState(TypedDict):
    input_file: Optional[str]  # For images
    input_csv: Optional[str]   # For CSV files
    messages: Annotated[List[AnyMessage], add_messages]

# Step 2: Preparing Tools
def extract_text(image_path: str) -> str:
    """
    Extract the text from the image using multimodal model
    """
    logger.info(f"Calling extract_text with image_path: {image_path}")
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "Error: Image file not found"

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            logger.error(f"Invalid or unsupported image format: {image_path}")
            return "Error: Invalid or unsupported image format"

        # Read image and encode as base64
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare the prompt
        message = [
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': "Extract all the text from this image. Return only the extracted text, no explanations",
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision model
        response = vision_model.invoke(message)
        logger.info("Text extracted successfully")
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def divide(a: int, b: int) -> float:
    """Divide a and b - for occasional math calculations"""
    logger.info(f"Calling divide with a={a}, b={b}")
    if b == 0:
        logger.error("Division by zero attempted")
        return "Error: Division by zero"
    return a / b

def analyze_csv(file_path: str, query: str) -> str:
    """
    Analyze a CSV file based on the user's query (e.g., calculate average, summarize data)
    """
    logger.info(f"Calling analyze_csv with file_path: {file_path}, query: {query}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return "Error: CSV file not found"
        
        # Validate CSV
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) == 0:
                logger.error(f"Invalid CSV: Empty or no columns in {file_path}")
                return "Error: Invalid CSV file (empty or no columns)"
        except Exception as e:
            logger.error(f"Invalid CSV format: {str(e)}")
            return f"Error: Invalid CSV format: {str(e)}"

        query_lower = query.lower()
        if "average" in query_lower or "mean" in query_lower:
            column = query.split("column")[-1].strip() if "column" in query_lower else ""
            if column in df.columns:
                return f"Average of {column}: {df[column].mean()}"
            return "Error: Column not found"
        elif "summarize" in query_lower or "summary" in query_lower:
            summary = df.describe().to_string()
            return f"Data Summary:\n{summary}"
        else:
            return "Error: Unsupported CSV query. Try 'average of column X' or 'summarize data'"
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}")
        return f"Error: {str(e)}"

# Add tools together
tools = [divide, extract_text, analyze_csv]
llm = ChatOpenAI(model='gpt-4o')
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# The Node
def assistant(state: AgentState):
    logger.info("Entering assistant node")
    textual_description_of_tool = """
    extract_text(image_path: str) -> str:
        Extract text from an image file.

    divide(a: int, b: int) -> float:
        Divide a and b.

    analyze_csv(file_path: str, query: str) -> str:
        Analyze a CSV file with queries like 'average of column X' or 'summarize data'.
    """

    image = state.get("input_file")
    csv_file = state.get("input_csv")
    messages = state.get("messages", [])

    # Validate inputs
    last_message = messages[-1].content.lower() if messages else ""
    if "extract_text" in last_message or "extract text" in last_message:
        if not image or not os.path.exists(image):
            logger.error(f"input_file is missing or invalid: {image}")
            return {
                "messages": messages + [SystemMessage(content="Error: No valid image file provided for text extraction")],
                "input_file": image,
                "input_csv": csv_file
            }
    if "analyze_csv" in last_message or "average" in last_message or "summarize" in last_message:
        if not csv_file or not os.path.exists(csv_file):
            logger.error(f"input_csv is missing or invalid: {csv_file}")
            return {
                "messages": messages + [SystemMessage(content="Error: No valid CSV file provided for analysis")],
                "input_file": image,
                "input_csv": csv_file
            }

    sys_msg = SystemMessage(
        content=f"""
        You are a helpful vision Agent named Alfred that serves Mr. Wayne and Batman.
        You can analyze documents, CSV files, and run computations with tools:

        {textual_description_of_tool}

        Currently loaded image: {image}
        Currently loaded CSV: {csv_file}
        """
    )

    try:
        response = llm_with_tools.invoke([sys_msg] + messages)
        logger.info("LLM response received")
        return {
            "messages": messages + [response],
            "input_file": image,
            "input_csv": csv_file
        }
    except Exception as e:
        logger.error(f"Error in assistant node: {str(e)}")
        return {
            "messages": messages + [SystemMessage(content=f"Error in assistant: {str(e)}")],
            "input_file": image,
            "input_csv": csv_file
        }

# The Graph
builder = StateGraph(AgentState)
builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    tools_condition,
    {'tools': 'tools', END: END}
)
builder.add_edge('tools', 'assistant')
react_graph = builder.compile()

# Export the compiled graph
__all__ = ['react_graph']