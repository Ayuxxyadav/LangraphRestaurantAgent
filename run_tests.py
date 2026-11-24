import sys
from restproj.agent import build_restaurant_graph, RestaurantState
from langchain_core.messages import HumanMessage

def test_case_1():
    print("\n" + "="*50)
    print("RUNNING TEST CASE 1")
    print("Scenario: Unrelated -> Partial Order -> Reject Partial -> Order Unavailable Dish -> End due to retry attempt exhausted")
    print("="*50)
    
    graph = build_restaurant_graph()
    state = RestaurantState()

    # Step 1: Unrelated question
    print("\n[Step 1] User: 'What is quantum computing?'")
    state.messages.append(HumanMessage(content="What is quantum computing?"))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}, Order retries remaining: {state.order_retry_count}")
    print(f"Agent: {state.messages[-1].content}")

    # Step 2: User orders partial order (paneer butter masala stock is 5, requests 15)
    print("\n[Step 2] User: 'I want 15 plates of paneer butter masala'")
    state.messages.append(HumanMessage(content="I want 15 plates of paneer butter masala"))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}, Available: {state.available_quantity}, Order retries: {state.order_retry_count}")
    print(f"Agent: {state.messages[-1].content}")

    # Step 3: User rejects the partial order and wants to order again
    print("\n[Step 3] User: 'No, I don't want 5 plates, I want to order something else.'")
    state.messages.append(HumanMessage(content="No, I don't want 5 plates, I want to order something else."))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}, Order retries: {state.order_retry_count}")
    print(f"Agent: {state.messages[-1].content}")

    # Step 4: User orders a dish which is not available in the menu (e.g. sushi)
    print("\n[Step 4] User: 'Give me 2 plates of sushi'")
    state.messages.append(HumanMessage(content="Give me 2 plates of sushi"))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}, Order retries: {state.order_retry_count}")
    print(f"Agent: {state.messages[-1].content}")

    # Step 5: Exhaust order retry if any remaining by ordering another unavailable item
    while state.order_retry_count > 0 and not state.final_result:
        print(f"\n[Exhausting remaining attempt] User: 'Give me 1 burger'")
        state.messages.append(HumanMessage(content="Give me 1 burger"))
        result = graph.invoke(state)
        state = RestaurantState(**result)
        print(f"Status: {state.status}, Order retries: {state.order_retry_count}")
        print(f"Agent: {state.messages[-1].content}")

    print(f"\nFinal State Result: {state.final_result}")
    passed = (state.order_retry_count <= 0 or "apology" in (state.final_result or "").lower() or state.status == "NOT_AVAILABLE")
    print(f"TEST CASE 1 RESULT: {'PASS' if passed else 'FAIL'}")
    return passed

def test_case_2():
    print("\n" + "="*50)
    print("RUNNING TEST CASE 2")
    print("Scenario: Order available -> Cook fails once -> Cook success -> Serve fails once -> Cook retry & serve success -> Complete")
    print("="*50)

    cook_outcomes = [False, True, True]  # 1st: fail, 2nd: success, 3rd (re-cook after serve fail): success
    serve_outcomes = [False, True]       # 1st: fail, 2nd: success

    def sim_cook():
        res = cook_outcomes.pop(0) if cook_outcomes else True
        print(f"   [Cook Attempt] -> {'SUCCESS' if res else 'FAIL'}")
        return res

    def sim_serve():
        res = serve_outcomes.pop(0) if serve_outcomes else True
        print(f"   [Serve Attempt] -> {'SUCCESS' if res else 'FAIL'}")
        return res

    graph = build_restaurant_graph(simulated_cook_fn=sim_cook, simulated_serve_fn=sim_serve)
    state = RestaurantState()

    print("\n[Step 1] User: 'I want 2 plates of dal makhani'")
    state.messages.append(HumanMessage(content="I want 2 plates of dal makhani"))
    result = graph.invoke(state)
    state = RestaurantState(**result)

    print(f"Final Status: {state.status}")
    print(f"Final Result: {state.final_result}")
    print(f"Last Agent Message: {state.messages[-1].content}")
    passed = state.status == "COMPLETE" and "completed successfully" in (state.final_result or "").lower()
    print(f"TEST CASE 2 RESULT: {'PASS' if passed else 'FAIL'}")
    return passed

def test_case_3():
    print("\n" + "="*50)
    print("RUNNING TEST CASE 3")
    print("Scenario: Partial order -> Reject -> Order available -> Cook fails -> Cook success -> Serve fails -> Cook retry success -> Serve fails again -> Cook retry exhausted -> FAIL")
    print("="*50)

    # In this scenario:
    # 1. Initial cook: fails (cook_retry_count becomes 1)
    # 2. Cook retry: success (status READY, cook_retry_count is 1)
    # 3. Serve: fails (serve_retry_count becomes 1, sends back to cook)
    # 4. Cook re-cook after serve fail: success (cook_retry_count decrements to 0)
    # 5. Serve: fails again (serve_retry_count becomes 0, or cook attempts exhausted) -> overall apology / FAIL
    cook_outcomes = [False, True, True]
    serve_outcomes = [False, False]

    def sim_cook():
        res = cook_outcomes.pop(0) if cook_outcomes else True
        print(f"   [Cook Attempt] -> {'SUCCESS' if res else 'FAIL'}")
        return res

    def sim_serve():
        res = serve_outcomes.pop(0) if serve_outcomes else False
        print(f"   [Serve Attempt] -> {'SUCCESS' if res else 'FAIL'}")
        return res

    graph = build_restaurant_graph(simulated_cook_fn=sim_cook, simulated_serve_fn=sim_serve)
    state = RestaurantState()

    # Step 1: user orders partial
    print("\n[Step 1] User: 'I want 12 plates of vegetable biryani'")
    state.messages.append(HumanMessage(content="I want 12 plates of vegetable biryani"))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}, Available: {state.available_quantity}")

    # Step 2: does not accept partial
    print("\n[Step 2] User: 'No, cancel that, I want to order something else.'")
    state.messages.append(HumanMessage(content="No, cancel that, I want to order something else."))
    result = graph.invoke(state)
    state = RestaurantState(**result)
    print(f"Status: {state.status}")

    # Step 3: order again fully available
    print("\n[Step 3] User: 'I want 2 plates of chicken tikka masala'")
    state.messages.append(HumanMessage(content="I want 2 plates of chicken tikka masala"))
    result = graph.invoke(state)
    state = RestaurantState(**result)

    print(f"Final Status: {state.status}")
    print(f"Cook retries remaining: {state.cook_retry_count}")
    print(f"Serve retries remaining: {state.serve_retry_count}")
    print(f"Final Result: {state.final_result}")
    print(f"Last Agent Message: {state.messages[-1].content}")

    passed = state.status != "COMPLETE" and ("apology" in (state.final_result or "").lower() or "failed" in (state.final_result or "").lower())
    print(f"TEST CASE 3 RESULT: {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == "__main__":
    r1 = test_case_1()
    r2 = test_case_2()
    r3 = test_case_3()
    print("\n" + "="*50)
    print(f"SUMMARY: Test 1: {'PASS' if r1 else 'FAIL'}, Test 2: {'PASS' if r2 else 'FAIL'}, Test 3: {'PASS' if r3 else 'FAIL'}")
    print("="*50)

