# 🍽️ Autonomous Restaurant Order Management Agent (LangGraph + Groq)

Welcome to the **Autonomous Restaurant Order Management System**! 

This project is a production-grade, state-machine-driven AI Agent built with **LangGraph**, **LangChain**, and **Groq (`openai/gpt-oss-120b`)**. It simulates end-to-end real-world operations of a restaurant—from understanding casual, multi-dish customer orders to stock verification, real-time kitchen cooking simulations with stochastic failure handling, table serving with recursive retry loops, and automated invoice calculation using mathematical tool calls.

If you are an engineer, interviewer, or tech enthusiast exploring this project, this guide explains every architectural decision, internal state transition, retry counter mechanism, and test verification in deep, conversational detail.

---

## 📌 Executive Summary (What does this project do?)

Traditional chatbots handle conversation linearly: you ask a question, they give an answer. But real-world business workflows are **stateful, non-linear, and failure-prone**:
1. Customers ask unrelated questions ("What's the weather?").
2. Inventory is finite (some items are in stock, some out of stock, some partially available).
3. Kitchens make mistakes (food can burn or fail preparation).
4. Waiters can drop a tray (serving can fail, requiring the food to be cooked again).
5. Math performed by raw LLMs can hallucinate (billing needs exact, deterministic arithmetic tools).

This project solves all of these challenges by modeling the restaurant pipeline as a **Directed Acyclic & Cyclic Graph (DAG + Feedback Loops)** using **LangGraph**.

```
                           +----------------------+
                           |     [User Input]     |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
              +----------->|   user_order Node    |<-----------+
              |            +----------+-----------+            |
              |                       |                        |
              |                       v                        |
              |            +----------------------+            |
              | (Retry/    |  order_confirm Node  |            |
              |  Partial)  +----------+-----------+            |
              +-----------------------+                        |
                                      | (Confirmed)            |
                                      v                        |
                           +----------------------+            |
              +----------->|      cook Node       |            |
              | (Retry     +----------+-----------+            |
              |  Fail)                | (Success: READY)       |
              +-----------------------+                        |
                                      |                        |
                                      v                        |
                           +----------------------+            |
                           |      serve Node      +------------+ (Serve Failed: Re-cook)
                           +----------+-----------+
                                      | (Served: Success)
                                      v
                           +----------------------+
                           |    gen_bill Node     | (Deterministic Math Tools)
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |    complete Node     |
                           +----------------------+
```

*(Note: At any stage, if retries are exhausted or input is rejected, the graph routes cleanly to the **`apology Node`** to issue a polite apology and safely end the conversation).*

---

## 🏗️ Deep Architectural Breakdown

### 1. The Core State (`RestaurantState`)

Every single node in LangGraph is a pure function that takes the current state, performs an action or LLM call, and returns updates to the state. 

Here is what the agent's memory looks like throughout the lifecycle:

```python
class RestaurantState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]  # Chat transcript
    items: List[OrderItem]                                # Multi-dish items requested
    dish_name: str                                        # Backward-compatible primary dish
    req_quantity: int                                     # Backward-compatible primary qty
    available_quantity: str                               # Stock info string
    status: str                                           # Global lifecycle state
    order_retry_count: int = 3                            # Max 3 invalid order attempts
    cook_retry_count: int = 2                             # Max 2 cooking attempts
    serve_retry_count: int = 2                            # Max 2 serving attempts
    total_bill: float = 0.0                               # Total calculated invoice amount
    final_result: Optional[str] = None                    # Terminal status summary
    next_step: Optional[str] = None                       # Explicit edge-routing indicator
```

---

## 🔄 Node-by-Node Working & Logic

### 1️⃣ `user_order` Node (Input & Intent Parsing)
- **Role:** Understands user speech using LLM structured extraction (`MultiOrderExtraction`).
- **Capabilities:**
  1. **Intent Guardrails:** If someone asks *"Who is the prime minister?"* or *"Help me write Python code"*, it flags `is_food_related=False`. It politely refuses (*"I am a restaurant ordering agent, not a general-purpose agent..."*) and reduces the `order_retry_count`.
  2. **Multi-Dish Extraction:** If someone inputs *"15 plate dal makhani and raita 10 plates"*, it parses:
     - `dal makhani` $\rightarrow$ 15
     - `raita` $\rightarrow$ 10
  3. **Handling Decisions:** Detects whether the user is answering *"yes, proceed with available"* or *"no, cancel and let me choose something else"*.

### 2️⃣ `order_confirm` Node (Inventory & Menu Validator)
- **Role:** Evaluates customer desires against hard business reality (the menu stock).
- **The Restaurant Menu:**
  - *Paneer Butter Masala* (Stock: 5, ₹350)
  - *Dal Makhani* (Stock: 10, ₹250)
  - *Vegetable Biryani* (Stock: 8, ₹300)
  - *Chicken Tikka Masala* (Stock: 7, ₹400)
  - *Garlic Naan* (Stock: 50, ₹50)
  - *Raita* (Stock: 20, ₹80)
  - *Green Salad* (Stock: 15, ₹100)
- **Three Core Decisions:**
  - **Full Available:** All items have `requested_qty <= stock`. Status $\rightarrow$ `CONFIRMED`. Routes to `cook`.
  - **Partially Available:** Dish is in menu, but `requested_qty > stock`. Pauses workflow, marks status `PARTIALLY_AVAILABLE`, and asks user if they want the partial amount.
  - **Not Available:** Dish is not in menu (e.g., *sushi*, *burger*), or stock is `0`, or quantity is $\le 0$. Decrements `order_retry_count`. If retries reach `0`, routes to `apology`.

### 3️⃣ `cook` Node (Kitchen Simulation with Stochastic Failures)
- **Role:** Represents kitchen food preparation.
- **Rules & Probabilities:**
  - Cooking is simulated with a **60% success rate** and a **40% failure rate** (`random.random() < 0.60`).
  - Cooking starts with `cook_retry_count = 2`.
  - If cooking fails: Decrements `cook_retry_count`.
    - If `cook_retry_count > 0`: Status becomes `COOK_FAILED` $\rightarrow$ Re-enters `cook` node for 1 retry.
    - If `cook_retry_count == 0`: Attempts exhausted $\rightarrow$ Routes to `apology` $\rightarrow$ Terminates gracefully.
  - If cooking succeeds: Status becomes `READY` $\rightarrow$ Routes to `serve` node.

### 4️⃣ `serve` Node (Table Service with Feedback to Kitchen)
- **Role:** Models table service by waiters.
- **Feedback Loop Mechanism:**
  - Starts with `serve_retry_count = 2`.
  - If serve succeeds: Status becomes `SERVED` $\rightarrow$ Routes to `gen_bill`.
  - **If serve fails (e.g., waiter spills food):**
    - Decrements `serve_retry_count`.
    - Because spoiled food cannot be served, the order **must be re-cooked!**
    - If both `serve_retry_count > 0` and `cook_retry_count > 0`, the graph creates a cyclic feedback edge back to `cook`!
    - If either retry count is exhausted, it issues an apology and terminates.

### 5️⃣ `gen_bill` Node (Deterministic Tool-Calling Calculator)
- **Role:** Generates an accurate, un-hallucinated invoice.
- **Why Tools?** LLMs are notorious for making basic multiplication and addition mistakes when calculating multi-item totals with taxes or discounts.
- **Implementation:**
  - We define 4 LangChain tools: `multiply`, `add`, `subtract`, `divide`.
  - `gen_bill` computes each line item using `multiply.invoke({"a": qty, "b": price})`.
  - It accumulates the total using `add.invoke({"a": current_total, "b": item_total})`.
  - Prints a clean, itemized invoice with the exact payable amount.

### 6️⃣ `apology` & `complete` Nodes (Clean Termination)
- Prevents infinite loops. Every path in the graph leads strictly to either `complete` (order served + billed) or `apology` (retries exhausted / request rejected), ending cleanly at `END`.

---

## 🎯 Detailed Retry Counter Mechanics

| Counter | Initial Value | Decrement Condition | Consequence when reaches 0 |
| :--- | :---: | :--- | :--- |
| **`order_retry_count`** | `3` | Unrelated question asked, or user orders dish not in menu, or invalid quantity. | Workflow stops accepting inputs. Apology issued. Routes to `END`. |
| **`cook_retry_count`** | `2` | Kitchen burns/fails the dish (40% probability). | Kitchen halts. Apology issued. Routes to `END`. |
| **`serve_retry_count`** | `2` | Waiter fails to serve dish. | Waitstaff halts. Food cannot be served. Apology issued. Routes to `END`. |

---

## 🧪 Real Test Scenarios & Verification

All three strict test scenarios specified in `prompt.md` were coded into a deterministic test harness (`run_tests.py`) and verified:

### 🔹 Test Case 1: The Indecisive / Invalid Customer
- **Flow:**
  1. User asks: *"What is quantum computing?"* $\rightarrow$ Intercepted as non-food. Retry counter drops to 2.
  2. User orders: *"15 plates of paneer butter masala"* $\rightarrow$ Stock is only 5. Agent flags `PARTIALLY_AVAILABLE`.
  3. User says: *"No, cancel that, I want to order something else."* $\rightarrow$ State transitions cleanly to `REJECTED_PARTIAL`.
  4. User orders: *"Give me 2 plates of sushi"* $\rightarrow$ Sushi is not on the menu. Counter drops to 1.
  5. User orders: *"Give me 1 burger"* $\rightarrow$ Burger is not on the menu. Counter drops to 0.
- **Outcome:** **PASS** (Terminated at `NOT_AVAILABLE` with apology; retry count exhausted).

### 🔹 Test Case 2: Kitchen Hiccup + Waiter Stumble $\rightarrow$ Happy Ending
- **Flow:**
  1. User orders: *"2 plates of dal makhani"* (Fully in stock: 10).
  2. Cook Attempt 1: Simulated `FAIL` $\rightarrow$ Cook Retry count drops to 1.
  3. Cook Attempt 2: Simulated `SUCCESS` $\rightarrow$ Status `READY`.
  4. Serve Attempt 1: Simulated `FAIL` $\rightarrow$ Serve Retry drops to 1. Sends food back to `cook`!
  5. Re-Cook Attempt: Simulated `SUCCESS` $\rightarrow$ Food prepared fresh again.
  6. Serve Attempt 2: Simulated `SUCCESS` $\rightarrow$ Status `SERVED`.
  7. `gen_bill`: Tool calculates $2 \times ₹250 = ₹500$.
- **Outcome:** **PASS** (Terminated at `COMPLETE` with bill invoice).

### 🔹 Test Case 3: Complete Operational Exhaustion
- **Flow:**
  1. User orders partial vegetable biryani, rejects it, and orders 2 chicken tikka masala.
  2. Cook fails once, retries, and succeeds. (Cook retries left: 1).
  3. Serve fails. Sent back to cook.
  4. Cook re-cooks successfully. (Cook retries left: 0).
  5. Serve fails a second time! Serve retries left: 0, and cook retries exhausted.
- **Outcome:** **PASS** (Terminated at `SERVE_FAILED` with apology).

---

## 🚀 How to Run the Project Locally

### Prerequisites
- Python 3.10+ (Recommended: Python 3.13 via `uv`)
- Groq API Key (with `Model_name = openai/gpt-oss-120b` or any standard Groq model)

### 1. Setup Environment
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone <repo-url>
cd restproj

# Install dependencies using uv (or standard pip)
uv sync
```

Ensure you have a `.env` file in the project root:
```env
GROQ_API_KEY="your-groq-api-key"
Model_name="openai/gpt-oss-120b"
```

### 2. Interactive CLI Mode
To talk directly with the restaurant agent:
```bash
uv run python -m restproj.agent
```
**Example conversation:**
```text
You: 15 plate dal makhani and raita 10 plates

Agent: 'dal makhani' is partially available. Requested: 15, Stock: 10. Would you like to proceed with the available quantity, or place a new order?

You: yes proceed with available

Agent: Great! Proceeding with your confirmed order of: 10 x dal makhani, 10 x raita.
Agent: Order confirmed: 10 x 'dal makhani', 10 x 'raita' in stock. Sending to the kitchen!
Agent: Kitchen update: Cooking of 10 x 'dal makhani', 10 x 'raita' is successful! Status is READY.
Agent: Your order of 10 x 'dal makhani', 10 x 'raita' has been served successfully! Enjoy your meal!
Agent: 
🧾 ---------- BILL INVOICE ----------
  • 10 x Dal Makhani @ ₹250 = ₹2500.00
  • 10 x Raita @ ₹80 = ₹800.00
------------------------------------
  Total Payable Amount: ₹3300.00
------------------------------------
Thank you for dining with us! Have a wonderful day!

[Finished]: Order completed successfully.
```

### 3. Automated Test Suite
To execute the comprehensive simulation test suite:
```bash
uv run python run_tests.py
```

---

## 🔮 Future Scope & Scalability

If this system were deployed in an enterprise restaurant chain (like McDonald's, Starbucks, or Haldirams), here is how the architecture is prepared to evolve:

1. **Persistent Memory & Session Resumption (LangGraph Checkpointers):**
   - Integrate `SqliteSaver` or `PostgresSaver` so customers ordering via WhatsApp or web can disconnect, reconnect, and have their order state preserved.
2. **Database & POS Integration:**
   - Replace the static in-memory `MENU` dictionary with live API / SQL calls to real POS inventory systems (e.g., Toast, Petpooja, Square).
3. **Dynamic Kitchen Display System (KDS) Webhooks:**
   - Instead of simulated `random.random()` probabilities, make `cook` an asynchronous human-in-the-loop node: the agent sends an MQTT/WebSocket event to the kitchen screen, and waits for chef confirmation before routing to `serve`.
4. **Payment Gateway Integration in `gen_bill`:**
   - Connect Stripe / Razorpay UPI QR code generation inside `gen_bill` so that customers receive a clickable payment link or dynamic QR code directly in chat.
5. **Multi-turn Customization & Modifiers:**
   - Support modifiers like *"Make it less spicy"*, *"No onions"*, or *"Add extra cheese"* with specialized state attributes.

---

## 💡 Why This Project Stands Out in System Design & Interviews

- **Determinism over Chaos:** LLMs are used solely where natural language understanding is required (intent & entity extraction). Business rules (inventory, state routing, retries, arithmetic) are handled deterministically in code.
- **Fail-Safe & Resilient:** Handles dropped plates, burnt food, invalid customer messages, and out-of-stock items without crashing or hallucinating.
- **Mathematical Accuracy:** Eliminates calculation hallucinations using native LangChain tools.
- **Observable Graph:** Built on LangGraph, making every step inspectable, debuggable, and extensible.

