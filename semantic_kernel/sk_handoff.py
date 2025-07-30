import os
from semantic_kernel import Kernel
from semantic_kernel.agents import HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from dotenv import load_dotenv

load_dotenv()


def get_agents() -> tuple[list, OrchestrationHandoffs]:
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )
    kernel.add_service(chat_service)

    triage = ChatCompletionAgent(
        name="TriageAdvisor",
        instructions=(
            "You are the Triage Advisor. "
            "Read customer queries, classify into equity or complex. "
            "Return the best advisor role to hand off to."
        ),
        service=chat_service,
    )
    equity = ChatCompletionAgent(
        name="EquitySpecialist",
        instructions=(
            "You are the Equity Specialist. "
            "Provide stock analysis, recommendation, and next steps for equity queries. "
            "Focus on price, valuation, and market catalysts."
        ),
        service=chat_service,
    )
    human = ChatCompletionAgent(
        name="HumanAdvisor",
        instructions=(
            "You are the Human Advisor. "
            "Handle questions that AI cannot. "
            "Provide clear, empathetic guidance or escalate as needed."
        ),
        service=chat_service,
    )

    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            "TriageAdvisor",
            {"EquitySpecialist": "Equity requests", "HumanAdvisor": "Complex cases"},
        )
        .add("EquitySpecialist", "TriageAdvisor", "Return to triage")
    )

    return [triage, equity, human], handoffs


async def run_handoff_investment_consultation():
    agents, handoffs = get_agents()
    orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=lambda m: print(f"{m.name}: {m.content}"),
        human_response_function=lambda: ChatMessageContent(
            role=AuthorRole.USER, content=input("Human: ")
        ),
    )
    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(task="What investment advice?", runtime=runtime)
    print(await result.get(timeout=120))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_handoff_investment_consultation())
