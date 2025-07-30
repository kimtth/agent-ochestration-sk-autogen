import asyncio
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ConcurrentOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentRequest:
    company: str
    financial_data: Dict[str, Any]
    request_id: str


async def run_concurrent_investment_analysis():
    """Run concurrent investment analysis scenario using Semantic Kernel orchestration"""
    investment_request = InvestmentRequest(
        company="TechCorp Inc.",
        financial_data={
            "revenue": "$10B",
            "profit_margin": "15%",
            "debt_to_equity": "0.3",
            "price_data": "Upward trend, RSI: 45, Moving averages bullish",
            "news_sentiment": "Recent product launch received positive reviews, strong Q3 earnings",
        },
        request_id="INV-2024-001",
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
            name="FundamentalAnalyst",
            instructions=(
                "You are a Fundamental Investment Analyst. "
                "Analyze financial statements, revenue growth, profitability ratios, debt levels, "
                "and valuation multiples. "
                "Provide your findings as a bullet list and conclude with BUY/HOLD/SELL."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="TechnicalAnalyst",
            instructions=(
                "You are a Technical Investment Analyst. "
                "Assess price charts, moving averages, RSI, MACD, and volume trends. "
                "Summarize your technical signals and conclude with BUY/HOLD/SELL."
            ),
            service=chat_service,
        ),
        ChatCompletionAgent(
            name="SentimentAnalyst",
            instructions=(
                "You are a Sentiment Investment Analyst. "
                "Evaluate recent news headlines, social media sentiment, and market mood. "
                "Provide sentiment score (positive/neutral/negative) and a recommendation."
            ),
            service=chat_service,
        ),
    ]

    # Orchestrate them concurrently
    orchestration = ConcurrentOrchestration(members=members)
    # Transform InvestmentRequest into a ChatMessageContent dict for the request body
    orchestration._input_transform = lambda task: {
        "role": "user",
        "content": json.dumps(asdict(task)),
    }
    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(task=investment_request, runtime=runtime)
    messages = await result.get(timeout=60)
    for m in messages:
        print(f"{m.name}: {m.content}")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(run_concurrent_investment_analysis())

"""
TechnicalAnalyst: **Technical Analysis of TechCorp Inc.**

1. **Price Trends**: The price data shows an upward trend, indicating positive momentum in the stock's movement. This suggests that market sentiment is currently leaning towards buying, which can be a bullish sign for investors.

2. **Moving Averages**: The mention of bullish moving averages indicates that short-term averages are likely above long-term averages. This further supports the idea of upward momentum and suggests that the stock may continue to perform well in the near term.

3. **RSI (Relative Strength Index)**: Currently at 45, which is below the neutral level of 50. A reading below 50 can indicate that the stock is neither overbought nor oversold, suggesting there may be more room for price appreciation without hitting overbought territory.

4. **MACD (Moving Average Convergence Divergence)**: Although specific MACD values were not provided, if we consider bullish moving averages in conjunction with the upward price trend, it is likely that the MACD would also reflect a favorable crossover or position.

5. **Volume Trends**: Although volume data was not explicitly provided, a sustained upward price movement ideally should be supported by increasing volume for confirmation of strength in that trend.

6. **Sentiment & News**: The recent positive sentiment from TechCorp Inc.'s product launch and strong Q3 earnings add an optimistic outlook for the stock. Strong fundamental performance often correlates with sustained technical strength.

**Conclusion**: Considering the upward price trend, bullish moving averages, mild RSI reading, and positive news sentiment, TechCorp Inc. exhibits strong signs of potential growth. Therefore, the recommendation is to **BUY**.
FundamentalAnalyst: **TechCorp Inc. Financial Analysis:**

- **Revenue**: TechCorp Inc. reported revenue of $10 billion, indicating a solid top line suggesting strong demand for its products.
- **Profitability Ratios**: With a profit margin of 15%, the company is maintaining a healthy level of profitability relative to its revenue, indicative of efficient cost management.
- **Debt Levels**: The debt-to-equity ratio of 0.3 suggests that TechCorp has a low level of debt relative to its equity, minimizing financial risk and indicating stability.
- **Market Sentiment**: The stock is trending upward, with a Relative Strength Index (RSI) of 45, indicating that it is neither overbought nor oversold, maintaining a neutral position for potential buyers. The moving averages are showing a bullish trend, reinforcing the upward momentum.
- **Recent Developments**: A recent product launch received positive reviews, which may bolster revenue and enhance brand image. Additionally, the strong Q3 earnings report reflects good overall financial health and investor sentiment.

**Conclusion**: Given TechCorp Inc.'s strong revenue, solid profit margins, low debt levels, and positive market sentiment reinforced by favorable news, it appears to be in a strong position.

**Recommendation: BUY**
SentimentAnalyst: **Sentiment Analysis for TechCorp Inc.**

1. **Financial Data Overview:**
   - **Revenue:** $10B
   - **Profit Margin:** 15%
   - **Debt to Equity Ratio:** 0.3 (indicative of low debt and strong financial health)
   - **Price Data:** Current price trend is upward with a Relative Strength Index (RSI) of 45, which suggests that the stock is currently neither overbought nor oversold. Moving averages are bullish, indicating positive momentum.

2. **Recent News Headlines:**
   - Recent product launch received positive reviews.
   - Strong Q3 earnings reports, indicating higher than expected profits which reflect positively on management and business operations.

3. **Social Media Sentiment:**
   - The recent product launch and earnings report have been well-received, as per social media discussions, with a generally optimistic tone from analysts and retail investors alike.

4. **Market Mood:**
   - The broader market sentiment is upbeat, driven by strong tech earnings and innovation in the sector, benefiting stocks like TechCorp Inc.

**Summary of Findings:**
The combination of solid financial metrics, positive news surrounding product launches, strong earnings, and favorable price trends suggests that TechCorp Inc. is in a good position to continue its upward trajectory. The positive reception of its recent product introduction supports a positive market outlook.

**Sentiment Score: Positive**

**Recommendation:**
Given the strong financial performance, positive news flow, and bullish technical indicators, I recommend a **Buy** investment strategy for TechCorp Inc. at this time. It is advisable to keep an eye on future product releases and earnings reports to adjust investment strategy as needed.
"""
