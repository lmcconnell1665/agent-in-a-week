from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SUPERVISOR_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a supervisor for an agent.
                
                You need to decide whether the agent should respond or clarify.
                If the agent should respond, you should return "respond".
                If the agent should clarify, you should return "clarify".

                CRITICAL: You MUST return a valid JSON object, not a markdown or text response.

                Example:
                {{
                    "decision": "respond"
                    "reasoning": "The user's question is clear and concise, so we can respond directly."
                }}
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

GENERATE_RESPONSE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful AI assistant tasked with answering the user's question.
                Make sure to refer to the user as {name}.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


GENERATE_CLARIFICATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful AI assistant tasked with clarifying the user's question.
                Ask the user a question to help clarify their request.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
