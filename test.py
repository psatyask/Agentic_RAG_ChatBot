
from agent import get_agent
from langchain_core.messages import HumanMessage
from database import init_db

init_db()

# Initialize the agent
agent = get_agent("gemini-3.8-flash")

thread_id = "thread-1"
# Thread configuration
config = {
    "configurable": {
        "thread_id": thread_id
    }
}

# Stream the agent response
for message_chunk, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(
                content="what is my name?"
            )
        ]
    },
    config=config,
    stream_mode="messages"):
    
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
