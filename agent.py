from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from db_utils import get_vector_index
from dotenv import load_dotenv
import asyncio

async def chat_loop(agent: ReActAgent):
    print("Welcome to the Resume Analysis Agent!")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in {"exit", "quit"}:
                break

            # Await the async chat call
            resp = await agent.run(user_input)

            # LlamaIndex responses often carry a .response text
            print(f"Agent: {getattr(resp, 'response', str(resp))}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    load_dotenv()
    # Initialize the Vector Store (and other settings)
    vector_index, _ = get_vector_index()
    
    # Create the Query Engine from the vector index
    query_engine = vector_index.as_query_engine(similarity_top_k=3)

    # Define the Retrieval Tool
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="resume_search",
            description=(
                "Useful for retrieving specific information about job candidates from their resumes."
                "Use this tool when the user asks about candidate skills, experience, or details."
            ),
        ),
    )

    # Initialize ReAct Agent
    agent = ReActAgent(
        tools=[query_engine_tool],
        verbose=True,
    )

    print("Welcome to the Resume Analysis Agent!")
    print("Type 'exit' or 'quit' to end the session.")

    await chat_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
