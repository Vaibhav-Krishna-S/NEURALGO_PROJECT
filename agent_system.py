import os
import json
import re
from openai import OpenAI

# Directly set the API key (replace with your actual OpenAI API key)
OPENAI_API_KEY = "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"

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

def get_flight_info(flight_number: str) -> dict:
    """
    Simulates flight data retrieval based on flight number.
    
    Args:
        flight_number (str): The flight number to query.
    
    Returns:
        dict: A dictionary containing flight details or an error message.
    """
    # Mock flight database
    flight_database = {
        "AI123": {
            "flight_number": "AI123",
            "departure_time": "08:00 AM",
            "destination": "Delhi",
            "status": "Delayed"
        },
        "AI456": {
            "flight_number": "AI456",
            "departure_time": "12:00 PM",
            "destination": "Mumbai",
            "status": "On Time"
        }
    }
    
    # Return flight info if exists, otherwise return an error
    return flight_database.get(flight_number, {"error": f"Flight {flight_number} not found in database."})


def info_agent_request(flight_number: str) -> str:
    """
    Calls get_flight_info and returns the flight data as a JSON string.
    
    Args:
        flight_number (str): The flight number to query.
    
    Returns:
        str: JSON string containing flight details or an error message.
    """
    flight_data = get_flight_info(flight_number)
    return json.dumps(flight_data)


def qa_agent_respond(user_query: str) -> str:
    """
    Processes user queries, extracts flight numbers, and calls the Info Agent.
    
    Args:
        user_query (str): The user's query.
    
    Returns:
        str: JSON string containing the answer or an error message.
    """
    # Extract flight number using regex
    match = re.search(r"Flight (\w+)", user_query)
    if not match:
        return json.dumps({"answer": "Flight number not found in query."})
    
    flight_number = match.group(1)
    flight_data = info_agent_request(flight_number)
    
    # Parse JSON response
    flight_info = json.loads(flight_data)
    if "error" in flight_info:
        return json.dumps({"answer": flight_info["error"]})
    
    # Format user-friendly response
    answer = (
        f"Flight {flight_info['flight_number']} departs at {flight_info['departure_time']} "
        f"to {flight_info['destination']}. Current status: {flight_info['status']}."
    )
    return json.dumps({"answer": answer})


def get_openai_response(prompt: str) -> str:
    """
    Sends a prompt to OpenAI's GPT model and returns the response.
    
    Args:
        prompt (str): The input prompt.
    
    Returns:
        str: The AI-generated response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "An error occurred while processing the request."


if __name__ == "__main__":
    # Test cases for Problem 1
    print("Test Case 1: get_flight_info('AI123')")
    print(get_flight_info("AI123"))  # Expected: Flight AI123 data
    print()

    print("Test Case 2: get_flight_info('AI999')")
    print(get_flight_info("AI999"))  # Expected: Error message
    print()

    print("Test Case 3: info_agent_request('AI123')")
    print(info_agent_request("AI123"))  # Expected: JSON string of AI123 data
    print()

    print("Test Case 4: qa_agent_respond('When does Flight AI123 depart?')")
    print(qa_agent_respond("When does Flight AI123 depart?"))  # Expected: User-friendly JSON response
    print()

    print("Test Case 5: qa_agent_respond('What is the status of Flight AI999?')")
    print(qa_agent_respond("What is the status of Flight AI999?"))  # Expected: Flight not found
    print()

    # Example usage of OpenAI integration
    print("Example Usage of OpenAI Integration:")
    print(get_openai_response("Explain how AI works."))

def translate_to_english(text: str) -> str:
    """
    Translates text to English using OpenAI.
    
    Args:
        text (str): The input text in any language.
    
    Returns:
        str: The translated text in English.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": f"Translate the following text to English: {text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error translating text: {e}")
        return "Translation failed."

# Modify qa_agent_respond to support multi-language
def qa_agent_respond(user_query: str) -> str:
    # Translate the query to English if needed
    translated_query = translate_to_english(user_query)
    print(f"Translated Query: {translated_query}")
    
    # Extract flight number using regex
    match = re.search(r"Flight (\w+)", translated_query)
    if not match:
        return json.dumps({"answer": "Flight number not found in query."})
    
    flight_number = match.group(1)
    flight_data = info_agent_request(flight_number)
    
    # Parse JSON response
    flight_info = json.loads(flight_data)
    if "error" in flight_info:
        return json.dumps({"answer": flight_info["error"]})
    
    # Format user-friendly response
    answer = (
        f"Flight {flight_info['flight_number']} departs at {flight_info['departure_time']} "
        f"to {flight_info['destination']}. Current status: {flight_info['status']}."
    )
    return json.dumps({"answer": answer})