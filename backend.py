# ============================================================
# TripC AI - Multi-Agent Travel Planner
# LangGraph + Groq + Tavily + Flight API + PostgreSQL
# ============================================================

import os
import uuid
import operator

import certifi
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# IMPORTS
# ============================================================

from typing import TypedDict, Annotated

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ============================================================
# DATABASE URL
# ============================================================

def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "\nDATABASE_URL is missing.\n"
            "Please add your Render PostgreSQL External Database URL "
            "to your .env file.\n"
        )

    # Remove accidental quotes
    database_url = database_url.strip().strip('"').strip("'")

    # Make sure SSL is enabled
    if "sslmode=" not in database_url.lower():
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


DATABASE_URL = get_database_url()


# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError(
        "\nGROQ_API_KEY is missing.\n"
        "Please add GROQ_API_KEY to your .env file.\n"
    )


if not GROQ_MODEL:
    raise ValueError(
        "GROQ_MODEL is missing. "
        "Please add GROQ_MODEL to your .env file."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0
)


# ============================================================
# STATE
# ============================================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int


# ============================================================
# FLIGHT AGENT
# ============================================================

def flight_agent(state: TravelState):

    print("\n✈️ Flight Agent started...")

    query = state["user_query"]

    try:

        flight_data = search_flights(query)

    except Exception as e:

        flight_data = (
            f"Flight search failed.\n"
            f"Reason: {str(e)}"
        )

    return {

        "flight_results": str(flight_data),

        "messages": [
            AIMessage(
                content="Flight results fetched."
            )
        ],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# HOTEL AGENT
# ============================================================

def hotel_agent(state: TravelState):

    print("\n🏨 Hotel Agent started...")

    query = (
        f"Best hotels for {state['user_query']}. "
        f"Include hotel name, location, approximate price, "
        f"rating and important facilities."
    )

    try:

        hotel_results = tavily_search(query)

    except Exception as e:

        hotel_results = (
            f"Hotel search failed.\n"
            f"Reason: {str(e)}"
        )

    return {

        "hotel_results": str(hotel_results),

        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    print("\n🗺️ Itinerary Agent started...")

    prompt = f"""
You are an expert international travel planner.

Create a complete practical travel itinerary.

USER REQUEST:
{state['user_query']}

FLIGHT RESULTS:
{state['flight_results']}

HOTEL RESULTS:
{state['hotel_results']}

Requirements:

1. Create a day-by-day itinerary.
2. Include sightseeing.
3. Include approximate travel time where useful.
4. Consider the user's budget.
5. Suggest practical transportation.
6. Suggest food/lunch/dinner options.
7. Do not invent exact flight prices if they are unavailable.
8. Clearly separate confirmed information from estimates.
9. Make the itinerary realistic and easy to follow.

Return a detailed itinerary.
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel planner."
                ),
                HumanMessage(
                    content=prompt
                )
            ]
        )

        itinerary = response.content

    except Exception as e:

        itinerary = (
            f"Itinerary generation failed.\n"
            f"Reason: {str(e)}"
        )

        response = AIMessage(content=itinerary)

    return {

        "itinerary": itinerary,

        "messages": [
            response
        ],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================

def final_agent(state: TravelState):

    print("\n🤖 Final Agent started...")

    final_prompt = f"""
Generate the final travel response for the user.

USER REQUEST:
{state['user_query']}

FLIGHTS:
{state['flight_results']}

HOTELS:
{state['hotel_results']}

ITINERARY:
{state['itinerary']}

Format the final answer using these sections:

# 1. Trip Summary

# 2. Flight Information

# 3. Hotel Suggestions

# 4. Day-by-Day Itinerary

# 5. Estimated Budget

# 6. Final Recommendations

Important:

- Be clear and practical.
- Respect the user's requested budget.
- Mention that live flight APIs may not provide ticket prices
  if pricing is unavailable.
- Do not claim that an estimated price is a confirmed price.
- Clearly identify estimated costs.
- Make the response useful for real travel planning.
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a professional AI travel "
                        "booking assistant."
                    )
                ),
                HumanMessage(
                    content=final_prompt
                )
            ]
        )

    except Exception as e:

        response = AIMessage(
            content=(
                "Final response generation failed.\n"
                f"Reason: {str(e)}"
            )
        )

    return {

        "messages": [
            response
        ],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(TravelState)


graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


# ============================================================
# GRAPH FLOW
# ============================================================

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


# ============================================================
# POSTGRESQL CONNECTION POOL
# ============================================================

print("\n🔌 Connecting to PostgreSQL...")


try:

    connection_pool = ConnectionPool(
        conninfo=DATABASE_URL,

        min_size=1,

        max_size=5,

        open=False,

        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },

        # Check connection before giving it to LangGraph
        check=ConnectionPool.check_connection,

        # Reconnect if connection is lost
        reconnect_timeout=30,

    )

    # Open the pool
    connection_pool.open()

    # Wait until at least one connection is available
    connection_pool.wait(timeout=30)

    print("✅ PostgreSQL connection pool created.")

except Exception as e:

    raise RuntimeError(
        "\n❌ PostgreSQL connection failed.\n"
        f"Reason: {str(e)}\n\n"
        "Check your DATABASE_URL in the .env file."
    )


# ============================================================
# POSTGRES CHECKPOINTER
# ============================================================

try:

    checkpointer = PostgresSaver(
        connection_pool
    )

    # Create LangGraph checkpoint tables
    checkpointer.setup()

    print("✅ LangGraph PostgreSQL checkpointer ready.")

except Exception as e:

    connection_pool.close()

    raise RuntimeError(
        "\n❌ LangGraph PostgreSQL checkpointer setup failed.\n"
        f"Reason: {str(e)}"
    )


# ============================================================
# COMPILE GRAPH
# ============================================================

travel_graph = graph.compile(
    checkpointer=checkpointer
)

print("✅ Travel graph compiled successfully.")


# ============================================================
# RUN TRAVEL AGENT
# ============================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    # --------------------------------------------------------
    # Validate user input
    # --------------------------------------------------------

    if not user_input or not user_input.strip():

        raise ValueError(
            "Travel request cannot be empty."
        )

    user_input = user_input.strip()


    # --------------------------------------------------------
    # Create thread ID
    # --------------------------------------------------------

    if not thread_id:

        thread_id = (
            f"user_{uuid.uuid4().hex}"
        )


    # --------------------------------------------------------
    # LangGraph configuration
    # --------------------------------------------------------

    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }


    # --------------------------------------------------------
    # Initial State
    # --------------------------------------------------------

    initial_state = {

        "messages": [
            HumanMessage(
                content=user_input
            )
        ],

        "user_query": user_input,

        "flight_results": "",

        "hotel_results": "",

        "itinerary": "",

        "llm_calls": 0
    }


    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    try:

        result = travel_graph.invoke(
            initial_state,
            config=config
        )

    except Exception as e:

        print("\n❌ Travel Agent Error:")
        print(str(e))

        raise RuntimeError(
            f"Travel agent execution failed: {str(e)}"
        ) from e


    # --------------------------------------------------------
    # Get final response
    # --------------------------------------------------------

    messages = result.get(
        "messages",
        []
    )


    if not messages:

        final_answer = (
            "No response was generated."
        )

    else:

        final_message = messages[-1]

        if hasattr(
            final_message,
            "content"
        ):

            final_answer = final_message.content

        else:

            final_answer = str(
                final_message
            )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "thread_id": thread_id,

        "answer": final_answer,

        "flight_results": result.get(
            "flight_results",
            ""
        ),

        "hotel_results": result.get(
            "hotel_results",
            ""
        ),

        "itinerary": result.get(
            "itinerary",
            ""
        ),

        "llm_calls": result.get(
            "llm_calls",
            0
        )
    }


# ============================================================
# OPTIONAL CLEANUP
# ============================================================

def close_database():

    global connection_pool

    try:

        if connection_pool:

            connection_pool.close()

            print(
                "\n🔌 PostgreSQL connection pool closed."
            )

    except Exception as e:

        print(
            f"Database close warning: {e}"
        )