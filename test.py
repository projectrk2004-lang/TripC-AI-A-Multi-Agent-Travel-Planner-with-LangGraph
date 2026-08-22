from backend import run_travel_agent


user_input = input(
    "Enter travel request: "
)


response = run_travel_agent(
    user_input
)


print("\n")
print("=" * 70)
print("FINAL TRAVEL PLAN")
print("=" * 70)

print(
    response["answer"]
)

print("\n")
print("=" * 70)
print("THREAD ID")
print("=" * 70)

print(
    response["thread_id"]
)

print("\n")
print("LLM Calls:",
      response["llm_calls"])