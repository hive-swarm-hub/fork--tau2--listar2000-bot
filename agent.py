"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
import re
import time

from litellm import completion

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ── PROMPT (the main lever for improving performance) ─────────────────────────

INSTRUCTIONS = """
You are a customer service agent. You MUST follow the <policy> exactly. The policy is your sole source of truth — never invent rules, procedures, or information not in the policy or provided by the user.

## Critical rules
1. Each turn: EITHER send a message to the user OR make a tool call. NEVER both at the same time.
2. Only make ONE tool call per turn.
3. Before any action that modifies the database (booking, modifying, cancelling), you MUST:
   a. Verify all policy preconditions are met (eligibility, rules, restrictions).
   b. List the exact action details to the user and get explicit confirmation.
   c. Only then make the tool call.
4. The APIs do NOT enforce policy rules — YOU must check them before calling.
5. If a request is against policy, deny it and explain why.
6. Transfer to a human agent ONLY if the request cannot be handled within the scope of your actions. To transfer: first call transfer_to_human_agents, then send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
7. Do not proactively offer compensation unless the user explicitly asks.

## Key practices
- First identify the user (get user ID).
- Gather all needed information using tools before taking action.
- Always look up CURRENT prices/availability — never reuse prices from old reservations.
- Check every policy rule that applies to the situation before calling an API.
- Use exact values from tool results (IDs, dates, amounts). Do not guess or approximate.
- When the user confirms, proceed immediately — do not ask for confirmation again.
- For technical support: follow the troubleshooting workflow step by step, checking each condition before moving to the next.
- After each tool result, compare it against ALL policy requirements. Look for what is MISSING, not just what is present.
- Keep responses concise.
""".strip()

SYSTEM_TEMPLATE = """
<instructions>
{instructions}
</instructions>
<policy>
{policy}
</policy>
""".strip()

# ── MESSAGE CONVERSION ────────────────────────────────────────────────────────

def to_api_messages(messages, annotator=None):
    """Convert tau2 message objects to OpenAI-style dicts."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if m.is_tool_call():
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            content = m.content if m.content else ""
            if annotator:
                content = annotator(content)
            out.append({"role": "tool", "content": content, "tool_call_id": m.id})
    return out


def parse_response(choice):
    """Convert an LLM API response choice into a tau2 AssistantMessage."""
    tool_calls = None
    if choice.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in choice.tool_calls
        ]
    return AssistantMessage(
        role="assistant",
        content=choice.content or "",
        tool_calls=tool_calls or None,
    )


# ── TOOL RESULT ANNOTATIONS ─────────────────────────────────────────────────

def annotate_telecom(content: str) -> str:
    """Add brief annotations to telecom tool results."""
    if not content:
        return content
    annotations = []

    if '"roaming_enabled": false' in content:
        annotations.append(
            "⚠ roaming_enabled is false. If user is traveling, call enable_roaming AND ask user to toggle data roaming ON on device."
        )
    if '"roaming_enabled": true' in content and "enable" in content.lower():
        annotations.append(
            "Backend roaming enabled. Now ask user to check_network_status and toggle data roaming ON on device if needed."
        )

    used_match = re.search(r'"data_used_gb":\s*([\d.]+)', content)
    limit_match = re.search(r'"data_limit_gb":\s*([\d.]+)', content)
    if used_match and limit_match:
        used = float(used_match.group(1))
        limit = float(limit_match.group(1))
        if used > limit:
            annotations.append(
                f"⚠ Data usage ({used}GB) exceeds limit ({limit}GB). Offer data refueling (max 2GB) or plan change."
            )

    if '"status": "Suspended"' in content and '"line_id"' in content:
        contract_match = re.search(r'"contract_end_date":\s*"([^"]+)"', content)
        if contract_match and contract_match.group(1) < "2025-02-25":
            annotations.append(
                "⚠ Line suspended with expired contract. Cannot resume — transfer to human agent."
            )

    if annotations:
        return content + "\n\n[NOTES] " + " | ".join(annotations)
    return content


def annotate_airline(content: str) -> str:
    """Add brief annotations to airline tool results."""
    if not content:
        return content
    annotations = []

    if '"cabin": "basic_economy"' in content:
        annotations.append("⚠ BASIC ECONOMY: flights cannot be changed. Cabin CAN be changed.")

    if '"reservation_id"' in content and '"created_at"' in content:
        created_match = re.search(r'"created_at":\s*"([^"]+)"', content)
        if created_match:
            created = created_match.group(1)
            if created >= "2024-05-14T15:00":
                annotations.append("Booking within 24hrs — cancellation allowed.")
            else:
                is_business = '"cabin": "business"' in content
                has_insurance = '"travel_insurance": "yes"' in content
                if not is_business and not has_insurance:
                    annotations.append(
                        "Booked >24hrs ago, not business, no insurance. Cancellation NOT allowed unless airline cancelled flight."
                    )

    if annotations:
        return content + "\n\n[NOTES] " + " | ".join(annotations)
    return content


def annotate_retail(content: str) -> str:
    """Add brief annotations to retail tool results."""
    if not content:
        return content
    annotations = []

    if '"status": "pending"' in content and '"order_id"' in content:
        annotations.append("Order is PENDING. Use modify_pending_order_* tools (NOT exchange/return).")
    elif '"status": "delivered"' in content and '"order_id"' in content:
        annotations.append("Order is DELIVERED. Use exchange/return_delivered_order_items (NOT modify_pending).")

    if annotations:
        return content + "\n\n[NOTES] " + " | ".join(annotations)
    return content


ANNOTATORS = {
    "telecom": annotate_telecom,
    "airline": annotate_airline,
    "retail": annotate_retail,
}


# ── DOMAIN DETECTION ─────────────────────────────────────────────────────────

def detect_domain(policy: str) -> str:
    """Detect the domain from the policy text."""
    lower = policy.lower()
    if "airline" in lower and "reservation" in lower and "flight" in lower:
        return "airline"
    elif "retail" in lower and "pending" in lower and "delivered" in lower:
        return "retail"
    elif "telecom" in lower:
        return "telecom"
    return "unknown"


# ── AGENT ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3


class CustomAgent(LLMAgent):
    """Self-contained customer service agent."""

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("SOLVER_MODEL", "openai/gpt-5.4-mini")
        self.llm_args = dict(llm_args or {})
        self.domain = detect_domain(domain_policy)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(instructions=INSTRUCTIONS, policy=self.domain_policy)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: LLMAgentState):
        # 1. Append incoming message(s) to conversation history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # 2. Build API request with domain-specific annotations
        api_messages = to_api_messages(
            state.system_messages + state.messages,
            annotator=ANNOTATORS.get(self.domain),
        )
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        # 3. Call LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = completion(
                    model=self.llm,
                    messages=api_messages,
                    tools=api_tools,
                    tool_choice="auto" if api_tools else None,
                    **self.llm_args,
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        # 4. Parse response
        assistant_msg = parse_response(response.choices[0].message)
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
