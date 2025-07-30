import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.agents import GroupChatOrchestration, RoundRobinGroupChatManager
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents.runtime import InProcessRuntime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentProposal:
    company: str
    proposal_details: Dict[str, Any]
    session_id: str


async def run_investment_committee_groupchat():
    """Run investment committee group chat using Semantic Kernel orchestration"""
    investment_proposal = InvestmentProposal(
        company="CleanEnergy Innovations Ltd.",
        proposal_details={
            "type": "Growth Equity Investment",
            "amount": "$50M",
            "sector": "Renewable Energy",
            "expected_return": "18-22% IRR",
            "risk_level": "Medium-High",
            "time_horizon": "5-7 years",
            "key_metrics": {
                "revenue_growth": "45% CAGR",
                "market_opportunity": "$500B by 2030",
                "competitive_position": "Patent-protected technology",
            },
        },
        session_id="COMMITTEE-2024-Q4-001",
    )

    # Initialize Semantic Kernel with Azure OpenAI
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )
    kernel.add_service(chat_service)

    # Create member agents (limit to 3)
    members = [
        ChatCompletionAgent(
            name="Moderator",
            description="Investment Committee Moderator agent",
            instructions=(
                "You are the Investment Committee Moderator. "
                "Manage turn order, prompt agents to speak, and keep discussion on track. "
                "After each round, summarize key points."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="FundamentalAnalyst",
            description="Agent analyzing financials and valuations",
            instructions=(
                "You are a Fundamental Analyst. "
                "Critically analyze company financials, valuation, and risks. "
                "Respond in 2–3 bullet points."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="RiskManager",
            description="Agent identifying and quantifying key risks",
            instructions=(
                "You are a Risk Manager. "
                "Identify and quantify key risks, mitigation strategies, and risk-adjusted returns. "
                "Respond in 2–3 bullet points."
            ),
            service=chat_service,
        ),
    ]

    orchestration = GroupChatOrchestration(
        members=members,
        manager=RoundRobinGroupChatManager(max_rounds=5),
        agent_response_callback=lambda m: print(f"{m.name}: {m.content}"),
    )
    # Transform InvestmentRequest into a ChatMessageContent dict for the request body
    orchestration._input_transform = lambda task: {
        "role": "user",
        "content": json.dumps(asdict(task)),
    }
    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(task=investment_proposal, runtime=runtime)
    decision = await result.get(timeout=60)
    print(decision)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_investment_committee_groupchat())

"""
Moderator: Thank you for the introduction of CleanEnergy Innovations Ltd. Let's get started by reviewing the proposal. 

I'll invite our first member, Agent A, to provide their initial thoughts on the investment.

Agent A, please share your insights.
FundamentalAnalyst: - **Financial Health and Growth Potential**: CleanEnergy Innovations Ltd. has an impressive revenue growth rate of 45% CAGR, presenting a solid case for scalability and market penetration. The projected market opportunity of $500 billion by 2030 indicates a substantial growth potential in the renewable energy sector, making this investment attractive for long-term returns.

- **Valuation and Expected Returns**: The proposal anticipates an IRR of 18-22%, which is relatively favorable given the medium-high risk level assessed. However, investors should cautiously consider if current valuations align with growth projections, especially in a highly competitive landscape where technological advancements can shift market dynamics rapidly.       

- **Risks and Competitive Landscape**: While the company boasts patent-protected technology, potential risks include regulatory changes affecting renewable energy subsidies, market volatility, and technological obsolescence. Valuation should also factor in competition from both established players and emerging startups that could challenge CleanEnergy's market share, warranting a thorough risk assessment before proceeding with the investment.
RiskManager: - **Key Risks**: The primary risks include regulatory changes impacting renewable energy policies, market competition from new technologies or incumbents, and operational risks associated with scaling production. To quantify, these could potentially reduce the expected IRR by 5-10% if adverse conditions arise.

- **Mitigation Strategies**: Establish strategic partnerships within the renewable sector to enhance market reach and innovation; conduct regular market analysis to stay ahead of regulatory shifts; and implement robust operational controls to manage growth efficiently. This can be supported by setting aside a reserve fund to address unexpected challenges.

- **Risk-Adjusted Returns**: Given the expected IRR of 18-22%, factoring in these risks and potential mitigations suggests a conservative risk-adjusted return in the range of 12-17%. This aligns with industry benchmarks for medium-high risk sector investments, making it a viable long-term investment opportunity.
Moderator: Thank you, Agent A and Risk Manager, for your detailed insights. 

**Key Points Summarized**:
- CleanEnergy Innovations Ltd. demonstrates significant growth potential with a 45% CAGR and a substantial market opportunity of $500 billion by 2030.
- The expected IRR of 18-22% is appealing, yet requires careful valuation assessment within a competitive and regulatory environment.
- Potential risks include regulatory changes, competition, and operational challenges, which could impact expected returns but can be mitigated through strategic partnerships and proactive market analysis.

Next, I invite Agent B to share their thoughts on the proposal. Agent B, please proceed.
FundamentalAnalyst: - **Market Positioning and Product Differentiation**: CleanEnergy’s competitive advantage lies in its patent-protected technology, which can provide a unique value proposition within the renewable energy sector. However, ongoing innovation and continuous improvement of this technology are crucial to maintain this edge against increasing competition. 

- **Investment Horizon and Liquidity Concerns**: The 5-7 year time horizon for returns could pose liquidity issues, especialntial exit strategies (e.g., IPO, acquisition) must be clearly defined and aligned with market conditions at the time. Unforeseen market downturns or shifts in investor sentiment regarding renewable energy could significantly impact exit valuations, necessitating a reassessment of anticipated returns.
- **Market Positioning and Product Differentiation**: CleanEnergy’s competitive advantage lies in its patent-protected technology, which can provide a unique value proposition within the renewable energy sector. However, ongoing innovation and continuous improvement of this technology are crucial to maintain this edge against increasing competition.

- **Investment Horizon and Liquidity Concerns**: The 5-7 year time horizon for returns could pose liquidity issues, especially if the company encounters unforeseen operational or market challenges, which are not uncommon in the renewable energy space. Investors should assess their liquidity needs against this timeline.

- **Exit Strategy and Market Exit Risks**: Given the rapid evolution of the renewable sector, potential exit strategies (e.g., IPO, acquisition) must be clearly defined and aligned with market conditions at the time. Unforeseen market downturns or shifts in investor sentiment regarding renewable energy could significantly impact exit valuations, necessitating a reassessment of anticipated returns.
"""