import os
import random
from typing import Annotated, Literal, Optional, Callable, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ==========================================
# 1. Menu Configuration
# ==========================================
MENU = {
    "paneer butter masala": {"price": 350, "stock": 5},
    "dal makhani": {"price": 250, "stock": 10},
    "vegetable biryani": {"price": 300, "stock": 8},
    "chicken tikka masala": {"price": 400, "stock": 7},
    "garlic naan": {"price": 50, "stock": 50},
    "raita": {"price": 80, "stock": 20},
    "green salad": {"price": 100, "stock": 15},
}

# ==========================================
# 2. Calculator Tools (Add, Subtract, Multiply, Divide)
# ==========================================
@tool
def add(a: float, b: float) -> float:
    """Adds two numbers: a + b."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtracts b from a: a - b."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers: a * b."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divides a by b: a / b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

CALCULATOR_TOOLS = [add, subtract, multiply, divide]

# ==========================================
# 3. State & Models Definition
# ==========================================
class OrderItem(BaseModel):
    dish_name: str
    req_quantity: int
    available_quantity: str = "0"
    status: str = "PENDING"  # AVAILABLE, PARTIALLY_AVAILABLE, NOT_AVAILABLE

class RestaurantState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    items: List[OrderItem] = Field(default_factory=list)
    dish_name: str = ""
    req_quantity: int = 0
    available_quantity: str = "0"
    status: str = "INIT"
    order_retry_count: int = 3
    cook_retry_count: int = 2
    serve_retry_count: int = 2
    total_bill: float = 0.0
    final_result: Optional[str] = None
    next_step: Optional[str] = None

class ExtractedItem(BaseModel):
    dish_name: str = Field(description="Name of the food item or dish.")
    quantity: int = Field(default=1, description="Quantity requested for this dish.")

class MultiOrderExtraction(BaseModel):
    is_food_related: bool = Field(
        description="True if user input is asking to order food or responding to an order question. False if completely unrelated."
    )
    is_accepting_partial: bool = Field(
        default=False,
        description="True if user confirms proceeding with partial available quantity."
    )
    is_rejecting_partial: bool = Field(
        default=False,
        description="True ONLY if user says 'no', 'cancel', or rejects partial order without placing a new order yet."
    )
    items: List[ExtractedItem] = Field(
        default_factory=list,
        description="List of dishes and their respective quantities extracted from the user input."
    )

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("Model_name", "openai/gpt-oss-120b").strip()
    return ChatGroq(model=model_name, api_key=api_key, temperature=0.0)

# ==========================================
# 4. Graph Nodes
# ==========================================

def user_order_node(state: RestaurantState) -> dict:
    last_user_msg = state.messages[-1].content if state.messages else ""
    llm = get_llm()
    structured_llm = llm.with_structured_output(MultiOrderExtraction)

    system_prompt = (
        "You are an AI order assistant for a restaurant.\n"
        "Menu items: " + ", ".join(MENU.keys()) + ".\n"
        "1. Identify if the user's message is related to ordering food. If unrelated (e.g. math, coding, general questions), set is_food_related=False.\n"
        "2. If user mentions one or more dishes (e.g. '15 plate dal makhani and raita 10 plates'), extract EACH dish and its respective quantity into the items list.\n"
        "3. If previous state had partial availability:\n"
        "   - If user confirms ('yes', 'proceed', 'go ahead with partial'), set is_accepting_partial=True.\n"
        "   - If user rejects ('no', 'cancel'), set is_rejecting_partial=True."
    )

    try:
        extraction: MultiOrderExtraction = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Current state: dishes={[it.dish_name for it in state.items]}, status={state.status}. User says: '{last_user_msg}'")
        ])
    except Exception:
        extraction = MultiOrderExtraction(is_food_related=True, items=[])

    # Case 1: Unrelated to food
    if not extraction.is_food_related:
        remaining_attempts = state.order_retry_count - 1
        msg = (
            "I am a restaurant ordering agent, not a general-purpose agent, and will not be able to help with that. "
            f"Please let me know what you would like to order from our menu! ({remaining_attempts} attempts remaining)"
        )
        return {
            "messages": [AIMessage(content=msg)],
            "status": "UNRELATED_INPUT",
            "next_step": "user_retry" if remaining_attempts > 0 else "apology",
            "order_retry_count": remaining_attempts
        }

    # Case 2: Accepting partial order
    if extraction.is_accepting_partial and state.status == "PARTIALLY_AVAILABLE":
        updated_items = []
        for it in state.items:
            avail = int(it.available_quantity)
            new_qty = min(it.req_quantity, avail) if avail > 0 else it.req_quantity
            updated_items.append(OrderItem(
                dish_name=it.dish_name,
                req_quantity=new_qty,
                available_quantity=it.available_quantity,
                status="AVAILABLE"
            ))
        summary = ", ".join([f"{it.req_quantity} x {it.dish_name}" for it in updated_items if it.req_quantity > 0])
        msg = f"Great! Proceeding with your confirmed order of: {summary}."
        first_dish = updated_items[0].dish_name if updated_items else ""
        first_qty = updated_items[0].req_quantity if updated_items else 0
        return {
            "messages": [AIMessage(content=msg)],
            "items": updated_items,
            "dish_name": first_dish,
            "req_quantity": first_qty,
            "status": "CONFIRMED",
            "next_step": "order_confirm"
        }

    # Case 3: Rejecting partial without specifying new order
    if extraction.is_rejecting_partial and not extraction.items:
        msg = "Understood. Please let me know what new dish and quantity you would like to order."
        return {
            "messages": [AIMessage(content=msg)],
            "status": "REJECTED_PARTIAL",
            "next_step": "user_retry"
        }

    # Case 4: New or updated items
    order_items: List[OrderItem] = []
    if extraction.items:
        for ext in extraction.items:
            d_name = ext.dish_name.lower().strip()
            for menu_dish in MENU:
                if menu_dish in d_name or d_name in menu_dish:
                    d_name = menu_dish
                    break
            qty = max(1, ext.quantity)
            order_items.append(OrderItem(dish_name=d_name, req_quantity=qty))
    else:
        order_items = state.items

    first_dish = order_items[0].dish_name if order_items else state.dish_name
    first_qty = order_items[0].req_quantity if order_items else state.req_quantity

    return {
        "items": order_items,
        "dish_name": first_dish,
        "req_quantity": first_qty,
        "next_step": "order_confirm"
    }


def order_confirm_node(state: RestaurantState) -> dict:
    if not state.items:
        items = [OrderItem(dish_name=state.dish_name, req_quantity=state.req_quantity)]
    else:
        items = state.items

    updated_items = []
    has_unavailable = False
    has_partial = False
    messages_out = []

    for it in items:
        dish = it.dish_name.lower().strip()
        req_qty = it.req_quantity

        if dish not in MENU:
            it_status = "NOT_AVAILABLE"
            avail = "0"
            has_unavailable = True
            messages_out.append(f"'{dish}' is not available in our menu (available: 0).")
        else:
            stock = MENU[dish]["stock"]
            avail = str(stock)
            if req_qty <= 0:
                it_status = "NOT_AVAILABLE"
                has_unavailable = True
                messages_out.append(f"Invalid quantity {req_qty} for '{dish}'.")
            elif stock == 0:
                it_status = "NOT_AVAILABLE"
                has_unavailable = True
                messages_out.append(f"'{dish}' is out of stock (available: 0).")
            elif req_qty > stock:
                it_status = "PARTIALLY_AVAILABLE"
                has_partial = True
                messages_out.append(f"'{dish}' is partially available. Requested: {req_qty}, Stock: {stock}.")
            else:
                it_status = "AVAILABLE"

        updated_items.append(OrderItem(
            dish_name=dish,
            req_quantity=req_qty,
            available_quantity=avail,
            status=it_status
        ))

    first_dish = updated_items[0].dish_name
    first_req = updated_items[0].req_quantity
    first_avail = updated_items[0].available_quantity

    if has_unavailable:
        remaining_attempts = state.order_retry_count - 1
        msg = " ".join(messages_out) + f" Please place a new order. ({remaining_attempts} order attempts remaining)"
        return {
            "items": updated_items,
            "dish_name": first_dish,
            "req_quantity": first_req,
            "available_quantity": first_avail,
            "status": "NOT_AVAILABLE",
            "order_retry_count": remaining_attempts,
            "messages": [AIMessage(content=msg)],
            "next_step": "user_retry" if remaining_attempts > 0 else "apology"
        }

    if has_partial:
        msg = " ".join(messages_out) + " Would you like to proceed with the available quantity, or place a new order?"
        return {
            "items": updated_items,
            "dish_name": first_dish,
            "req_quantity": first_req,
            "available_quantity": first_avail,
            "status": "PARTIALLY_AVAILABLE",
            "messages": [AIMessage(content=msg)],
            "next_step": "user_retry"
        }

    summary = ", ".join([f"{it.req_quantity} x '{it.dish_name}'" for it in updated_items])
    msg = f"Order confirmed: {summary} in stock. Sending to the kitchen!"
    return {
        "items": updated_items,
        "dish_name": first_dish,
        "req_quantity": first_req,
        "available_quantity": first_avail,
        "status": "CONFIRMED",
        "messages": [AIMessage(content=msg)],
        "next_step": "cook"
    }


def create_cook_node(simulated_fn: Optional[Callable[[], bool]] = None):
    def cook_node(state: RestaurantState) -> dict:
        if simulated_fn:
            success = simulated_fn()
        else:
            success = random.random() < 0.60

        summary = ", ".join([f"{it.req_quantity} x '{it.dish_name}'" for it in state.items]) if state.items else f"{state.req_quantity} x '{state.dish_name}'"

        if success:
            msg = f"Kitchen update: Cooking of {summary} is successful! Status is READY."
            return {
                "status": "READY",
                "messages": [AIMessage(content=msg)],
                "next_step": "serve"
            }
        else:
            new_attempts = state.cook_retry_count - 1
            if new_attempts > 0:
                msg = f"Kitchen update: Cooking failed. Retrying... ({new_attempts} cook attempt(s) remaining)."
                return {
                    "status": "COOK_FAILED",
                    "cook_retry_count": new_attempts,
                    "messages": [AIMessage(content=msg)],
                    "next_step": "cook"
                }
            else:
                msg = f"Kitchen update: Cooking failed and all cook attempts have been exhausted."
                return {
                    "status": "COOK_FAILED",
                    "cook_retry_count": 0,
                    "messages": [AIMessage(content=msg)],
                    "next_step": "apology"
                }
    return cook_node


def create_serve_node(simulated_fn: Optional[Callable[[], bool]] = None):
    def serve_node(state: RestaurantState) -> dict:
        if simulated_fn:
            success = simulated_fn()
        else:
            success = random.random() < 0.70

        summary = ", ".join([f"{it.req_quantity} x '{it.dish_name}'" for it in state.items]) if state.items else f"{state.req_quantity} x '{state.dish_name}'"

        if success:
            msg = f"Your order of {summary} has been served successfully! Enjoy your meal!"
            return {
                "status": "SERVED",
                "messages": [AIMessage(content=msg)],
                "next_step": "gen_bill"
            }
        else:
            new_serve_attempts = state.serve_retry_count - 1
            if new_serve_attempts > 0 and state.cook_retry_count > 0:
                msg = (
                    f"Serve failed! Sending back to kitchen to cook again. "
                    f"({new_serve_attempts} serve attempt(s) left, {state.cook_retry_count} cook attempt(s) left)."
                )
                return {
                    "status": "SERVE_FAILED",
                    "serve_retry_count": new_serve_attempts,
                    "messages": [AIMessage(content=msg)],
                    "next_step": "cook"
                }
            else:
                reason = "Serve failed multiple times" if new_serve_attempts == 0 else "Cook retry attempts exhausted after serve failure"
                msg = f"Order could not be served ({reason})."
                return {
                    "status": "SERVE_FAILED",
                    "serve_retry_count": max(0, new_serve_attempts),
                    "messages": [AIMessage(content=msg)],
                    "next_step": "apology"
                }
    return serve_node

cook_node = create_cook_node()
serve_node = create_serve_node()


def gen_bill_node(state: RestaurantState) -> dict:
    """
    Called when serve node succeeds.
    Uses calculator tools (multiply and add) to compute exact bill amount for items.
    """
    items = state.items if state.items else [OrderItem(dish_name=state.dish_name, req_quantity=state.req_quantity)]

    bill_lines = []
    total_amount = 0.0

    for it in items:
        dish = it.dish_name.lower().strip()
        price = MENU.get(dish, {}).get("price", 0.0)
        qty = float(it.req_quantity)
        
        # Tool call: multiply(qty, price)
        item_total = multiply.invoke({"a": qty, "b": float(price)})
        # Tool call: add(total_amount, item_total)
        total_amount = add.invoke({"a": total_amount, "b": item_total})
        
        bill_lines.append(f"  • {int(qty)} x {dish.title()} @ ₹{price} = ₹{item_total:.2f}")

    bill_text = (
        "\n🧾 ---------- BILL INVOICE ----------\n"
        + "\n".join(bill_lines) + "\n"
        + "------------------------------------\n"
        + f"  Total Payable Amount: ₹{total_amount:.2f}\n"
        + "------------------------------------\n"
        + "Thank you for dining with us! Have a wonderful day!"
    )

    return {
        "status": "COMPLETE",
        "total_bill": total_amount,
        "final_result": "Order completed successfully.",
        "messages": [AIMessage(content=bill_text)],
        "next_step": "complete"
    }


def apology_node(state: RestaurantState) -> dict:
    summary = ", ".join([it.dish_name for it in state.items]) if state.items else state.dish_name
    msg = (
        f"We sincerely apologize, but we are unable to complete your order for '{summary}'. "
        "Our team regrets the inconvenience and hopes to serve you better next time."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "final_result": f"Order failed. Apology issued. Status: {state.status}",
        "next_step": "end"
    }

def complete_node(state: RestaurantState) -> dict:
    return {
        "final_result": "Order completed successfully.",
        "next_step": "end"
    }

# ==========================================
# 5. Routing Logic
# ==========================================

def route_order(state: RestaurantState) -> Literal["cook", "user_retry", "apology"]:
    if state.next_step == "cook":
        return "cook"
    elif state.next_step == "apology":
        return "apology"
    else:
        return "user_retry"

def route_cook(state: RestaurantState) -> Literal["cook", "serve", "apology"]:
    if state.next_step == "serve":
        return "serve"
    elif state.next_step == "cook":
        return "cook"
    else:
        return "apology"

def route_serve(state: RestaurantState) -> Literal["cook", "gen_bill", "apology"]:
    if state.next_step == "gen_bill":
        return "gen_bill"
    elif state.next_step == "cook":
        return "cook"
    else:
        return "apology"

# ==========================================
# 6. Build Graph
# ==========================================

def build_restaurant_graph(simulated_cook_fn=None, simulated_serve_fn=None):
    builder = StateGraph(RestaurantState)

    builder.add_node("user_order", user_order_node)
    builder.add_node("order_confirm", order_confirm_node)
    builder.add_node("cook", create_cook_node(simulated_cook_fn))
    builder.add_node("serve", create_serve_node(simulated_serve_fn))
    builder.add_node("gen_bill", gen_bill_node)
    builder.add_node("apology", apology_node)
    builder.add_node("complete", complete_node)

    builder.add_edge(START, "user_order")

    builder.add_conditional_edges(
        "user_order",
        lambda s: "order_confirm" if s.next_step == "order_confirm" else (
            "apology" if s.order_retry_count <= 0 else END
        ),
        {"order_confirm": "order_confirm", "apology": "apology", END: END}
    )

    builder.add_conditional_edges(
        "order_confirm",
        route_order,
        {"cook": "cook", "user_retry": END, "apology": "apology"}
    )

    builder.add_conditional_edges(
        "cook",
        route_cook,
        {"cook": "cook", "serve": "serve", "apology": "apology"}
    )

    builder.add_conditional_edges(
        "serve",
        route_serve,
        {"cook": "cook", "gen_bill": "gen_bill", "apology": "apology"}
    )

    builder.add_edge("gen_bill", "complete")
    builder.add_edge("apology", END)
    builder.add_edge("complete", END)

    return builder.compile()

# Interactive CLI runner
def run_interactive():
    graph = build_restaurant_graph()
    state = RestaurantState()

    print("\n" + "="*50)
    print("Welcome to the LangGraph Restaurant Ordering System!")
    print("="*50)
    print("Menu:")
    for item, details in MENU.items():
        print(f" - {item.title()}: ₹{details['price']} (Stock: {details['stock']})")
    print("="*50 + "\n")

    while state.final_result is None:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        prev_msg_count = len(state.messages)
        state.messages.append(HumanMessage(content=user_input))
        
        # Invoke graph
        result = graph.invoke(state)
        state = RestaurantState(**result)
        
        # Print newly added AI messages
        new_msgs = state.messages[prev_msg_count + 1:]
        for msg in new_msgs:
            if isinstance(msg, AIMessage):
                print(f"\nAgent: {msg.content}")

        if state.final_result:
            print(f"\n[Finished]: {state.final_result}")
            break

        if state.order_retry_count <= 0:
            print("\nOrder retry limit exceeded.")
            break

if __name__ == "__main__":
    run_interactive()
