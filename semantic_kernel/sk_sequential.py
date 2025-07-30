import json
import os
from dataclasses import asdict, dataclass
from typing import List, Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.agents import SequentialOrchestration
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.agents.runtime import InProcessRuntime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentTask:
    company: str
    initial_data: Dict[str, Any]
    analysis_chain: List[str] = None


async def run_sequential_investment_analysis():
    """Run sequential investment analysis using Semantic Kernel orchestration"""
    investment_task = InvestmentTask(
        company="GreenTech Solutions Inc.",
        initial_data={
            "sector": "Renewable Energy",
            "market_cap": "$5B",
            "recent_price": "$45.20",
            "analyst_coverage": "Strong",
            "news_flow": "Positive - major contract wins",
        },
    )

    # Initialize Semantic Kernel with Azure OpenAI
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )
    kernel.add_service(chat_service)

    members = [
        ChatCompletionAgent(
            name="DataCollector",
            instructions=(
                "You are an Investment Data Collector. "
                "Given a company name and initial briefing, gather, verify, and structure key financial metrics: "
                "revenue, profit margin, debt ratios, recent price trends, analyst coverage, and relevant news. "
                "Output as a JSON-like object or bullet list with metric names and values."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="FundamentalAnalyst",
            instructions=(
                "You are a Fundamental Investment Analyst. "
                "Using the collected data, analyze financial health, business model strength, competitive position, "
                "growth prospects, and valuation. Identify key risks and opportunities. "
                "Conclude with a recommendation (BUY/HOLD/SELL) and a brief rationale (1–2 sentences)."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="ReportGenerator",
            instructions=(
                "You are an Investment Report Generator. "
                "Compose a concise executive summary of the analysis. "
                "List key findings as bullet points, state the final recommendation clearly, "
                "and provide supporting rationale in professional language."
            ),
            service=chat_service,
        ),
    ]

    orchestration = SequentialOrchestration(
        members=members,
        agent_response_callback=lambda m: print(f"{m.name}: {m.content}"),
    )
    # Transform InvestmentQuery into a ChatMessageContent dict for the request body
    orchestration._input_transform = lambda task: ChatMessageContent(
        role=AuthorRole.USER,
        content=json.dumps(asdict(task)),
    )
    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(task=investment_task, runtime=runtime)
    print(await result.get(timeout=60))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_sequential_investment_analysis())

"""
DataCollector: {
  "company": "GreenTech Solutions Inc.",
  "sector": "Renewable Energy",
  "financial_metrics": {
    "market_cap": "$5B",
    "recent_price": "$45.20",
    "revenue": {
      "value": "$1.2B",
      "currency": "USD"
    },
    "profit_margin": {
      "value": "15%",
      "currency": "Percentage"
    },
    "debt_ratios": {
      "debt_to_equity": "0.5",
      "current_ratio": "2.0"
    },
    "price_trend": {
      "last_month_change": "10%",
      "last_quarter_change": "5%"
    }
  },
  "analyst_coverage": {
    "rating": "Strong",
    "number_of_analysts": 12,
    "average_target_price": "$50.00"
  },
  "news_flow": {
    "general_sentiment": "Positive",
    "recent_developments": "Major contract wins in wind energy sector"
  }
}
FundamentalAnalyst: **Company Analysis: GreenTech Solutions Inc.**

**Financial Health:**
GreenTech Solutions Inc. demonstrates a robust financial position with a market capitalization of $5 billion and a solid revenue of $1.2 billion. The company exhibits a healthy profit margin of 15%, indicating effective cost management and operational efficiency. Additionally, the debt-to-equity ratio of 0.5 points to moderate leverage, while a current ratio of 2.0 reflects excellent liquidity, allowing the company to cover short-term liabilities comfortably.

**Business Model Strength:**
GreenTech operates in the renewable energy sector, focusing on innovative solutions for wind energy. The recent major contract wins underscore the strength of its business model and its ability to secure substantial revenue streams moving forward. The company is well-positioned to capitalize on the global shift towards sustainable energy practices.

**Competitive Position:**
With positive analyst coverage (rating: Strong from 12 analysts and an average target price of $50), GreenTech Solutions Inc. is recognized as a key player in its niche. Its recent developments, particularly in acquiring contracts, enhance its competitive position against peers in the renewable energy market.

**Growth Prospects:**
The overall market sentiment for GreenTech Solutions is positive, especially with burgeoning demand in the renewable energy sector. The company’s recent contract wins indicate strong revenue growth potential, aligning with trends favoring sustainable energy solutions.

**Valuation:**
Currently trading at $45.20, with analysts projecting a target price of $50, the stock shows room for appreciation. Given the positive outlook and strong financial metrics, the valuation appears attractive relative to projected growth.

**Key Risks and Opportunities:**
Key risks include potential regulatory changes or competition from larger corporations entering the renewable space; however, opportunities abound through strategic partnerships, technological advancements, and growing worldwide emphasis on climate change responsiveness.

**Recommendation:** **BUY**

**Rationale:** Given GreenTech Solutions Inc.'s solid financial health, robust growth prospects, and favorable market conditions coupled with a positive analyst sentiment, it represents a compelling investment opportunity with potential upside in share price.
ReportGenerator: **Executive Summary: Investment Analysis of GreenTech Solutions Inc.**

**Key Findings:**
- **Strong Financial Position:** Market cap of $5 billion with $1.2 billion in revenue and a 15% profit margin.
- **Moderate Leverage:** A debt-to-equity ratio of 0.5 indicates manageable debt levels.
- **Excellent Liquidity:** A current ratio of 2.0 ensures short-term financial obligations can be met.
- **Resilient Business Model:** Positioned well in the renewable energy sector with contracts enhancing revenue.
- **Favorable Analyst Sentiment:** "Strong" ratings from 12 analysts and a target price of $50, indicating growth potential.
- **Positive Market Outlook:** Significant demand for sustainable energy solutions aligns with the company's strategy.      
- **Valuation Opportunity:** Current trading at $45.20 with a projected gain based on target prices.

**Final Recommendation:** **BUY**

**Supporting Rationale:**
GreenTech Solutions Inc. offers a compelling investment proposition characterized by a sound financial foundation, an established footprint in the growing renewable energy market, and positive market dynamics. With robust revenue growth potential driven by recent contract acquisitions and favorable analyst sentiment, the stock is well-positioned for appreciation, underlining the recommendation to purchase.
**Executive Summary: Investment Analysis of GreenTech Solutions Inc.**

**Key Findings:**
- **Strong Financial Position:** Market cap of $5 billion with $1.2 billion in revenue and a 15% profit margin.
- **Moderate Leverage:** A debt-to-equity ratio of 0.5 indicates manageable debt levels.
- **Excellent Liquidity:** A current ratio of 2.0 ensures short-term financial obligations can be met.
- **Resilient Business Model:** Positioned well in the renewable energy sector with contracts enhancing revenue.
- **Favorable Analyst Sentiment:** "Strong" ratings from 12 analysts and a target price of $50, indicating growth potential.
- **Positive Market Outlook:** Significant demand for sustainable energy solutions aligns with the company's strategy.      
- **Valuation Opportunity:** Current trading at $45.20 with a projected gain based on target prices.

**Final Recommendation:** **BUY**

**Supporting Rationale:**
GreenTech Solutions Inc. offers a compelling investment proposition characterized by a sound financial foundation, an established footprint in the growing renewable energy market, and positive market dynamics. With robust revenue growth potential driven by recent contract acquisitions and favorable analyst sentiment, the stock is well-positioned for appreciation, underlining the recommendation to purchase.
"""
