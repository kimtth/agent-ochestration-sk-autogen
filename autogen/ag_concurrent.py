import asyncio
import os
from dataclasses import dataclass
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    default_subscription,
    message_handler,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentRequest:
    company: str
    financial_data: dict
    request_id: str


@dataclass
class AnalysisResult:
    analyst_type: str
    company: str
    analysis: str
    recommendation: str
    confidence_score: float
    request_id: str


RESULTS_TOPIC_TYPE = "analysis-results"
results_topic_id = TopicId(type=RESULTS_TOPIC_TYPE, source="default")


# --- new common base class ---
class BaseAnalysisAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        agent_name: str,
        service_id: str,
        system_prompt: str,
    ) -> None:
        super().__init__(agent_name)
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_prompt)
        self._service_id = service_id

    @message_handler
    async def analyze(self, message: InvestmentRequest, ctx: MessageContext) -> None:
        prompt = (
            f"Task for {self._service_id.replace('_',' ').title()}:\n"
            f"Company: {message.company}\n"
            f"Data: {message.financial_data}"
        )
        result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source="user")],
            cancellation_token=ctx.cancellation_token,
        )
        content = result.content
        assert isinstance(content, str)
        print(f"\n{'-'*60}\n{self.id.type}:\n{content}")

        # simple recommendation extraction
        recommendation = "HOLD"
        confidence = 0.7
        if "BUY" in content.upper():
            recommendation, confidence = "BUY", 0.8
        elif "SELL" in content.upper():
            recommendation, confidence = "SELL", 0.75

        analysis = AnalysisResult(
            analyst_type=self._service_id,
            company=message.company,
            analysis=content,
            recommendation=recommendation,
            confidence_score=confidence,
            request_id=message.request_id,
        )
        await self.publish_message(analysis, topic_id=results_topic_id)


@default_subscription
class FundamentalAnalyst(BaseAnalysisAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__(
            model_client,
            agent_name="Fundamental Analysis Agent",
            service_id="fundamental_analyst",
            system_prompt=(
                "You are a fundamental analyst specializing in company valuation. "
                "Analyze financial statements, revenue growth, and market position. "
                "Provide bullet-point findings and BUY/HOLD/SELL recommendation."
            ),
        )


@default_subscription
class TechnicalAnalyst(BaseAnalysisAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__(
            model_client,
            agent_name="Technical Analysis Agent",
            service_id="technical_analyst",
            system_prompt=(
                "You are a technical analyst specializing in price patterns and volume trends. "
                "Assess moving averages, RSI, MACD. Provide bullet-point summary and recommendation."
            ),
        )


@default_subscription
class SentimentAnalyst(BaseAnalysisAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__(
            model_client,
            agent_name="Sentiment Analysis Agent",
            service_id="sentiment_analyst",
            system_prompt=(
                "You are a sentiment analyst focusing on news and social media trends. "
                "Evaluate sentiment, provide a score, and BUY/HOLD/SELL recommendation."
            ),
        )


async def run_concurrent_investment_analysis():
    """Run concurrent investment analysis scenario"""

    # Sample investment data
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

    # Setup runtime and Azure OpenAI client
    runtime = SingleThreadedAgentRuntime()
    client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="gpt-4o-mini" # a name of OpenAI model (not clear meaning of this value when using Azure OpenAI)
    )

    # Register 3 analysts
    await FundamentalAnalyst.register(
        runtime, "fundamental_analyst", lambda: FundamentalAnalyst(client)
    )
    await TechnicalAnalyst.register(
        runtime, "technical_analyst", lambda: TechnicalAnalyst(client)
    )
    await SentimentAnalyst.register(
        runtime, "sentiment_analyst", lambda: SentimentAnalyst(client)
    )

    # Start analysis
    runtime.start()
    print(
        f"Starting concurrent investment analysis for {investment_request.company}..."
    )

    # Publish to all analysts
    await runtime.publish_message(investment_request, topic_id=DefaultTopicId())

    # Wait for completion
    await runtime.stop_when_idle()
    await client.close()


if __name__ == "__main__":
    asyncio.run(run_concurrent_investment_analysis())

"""
Starting concurrent investment analysis for TechCorp Inc....

------------------------------------------------------------
sentiment_analyst:
**Sentiment Evaluation for TechCorp Inc.:**

1. **Financial Indicators:**
   - **Revenue**: $10 billion is a strong indicator of company size and market presence.
   - **Profit Margin**: 15% suggests profitability and efficiency in operations, which is a positive signal.
   - **Debt-to-Equity Ratio**: A ratio of 0.3 indicates that the company is not overly reliant on debt to finance its operations, showing financial stability and lower risk.

2. **Market Performance:**
   - **Price Data**: An upward trend in stock price, with a Relative Strength Index (RSI) of 45 (indicating neither overbought nor oversold conditions), and bullish moving averages suggest positive market momentum and investor confidence.

3. **News Sentiment:**
   - The recent product launch receiving positive reviews signifies strong consumer interest and potential for future sales. 
   - Strong Q3 earnings further highlight the company's robust performance and ability to generate profits.

**Overall Sentiment Analysis:**
The sentiment around TechCorp Inc. appears overwhelmingly positive based on the financial health, market indicators, and news coverage. This suggests that the company is well-positioned for growth, making it an attractive investment opportunity.

**Sentiment Score:**
On a scale of -100 to +100, I would assign a sentiment score of +75.

**Recommendation:**
**BUY** - Given the positive financial metrics, favorable market indicators, and strong news sentiment, TechCorp Inc. demonstrates strong potential for future performance. It would be prudent to invest in the company at this time.

------------------------------------------------------------
technical_analyst:
### Technical Analysis Summary for TechCorp Inc.

- **Price Trends**:
  - **Upward Trend**: The price is currently exhibiting an upward trajectory, implying increased investor confidence and demand for the stock.  

- **Relative Strength Index (RSI)**:
  - **RSI: 45**: This suggests that the stock is neither overbought nor oversold, indicating potential for upward momentum without immediate risk of a pullback.

- **Moving Averages**:
  - **Bullish Moving Averages**: The presence of bullish moving averages signifies a positive long-term sentiment regarding the stock's performance. This typically indicates that the stock price is expected to continue rising.

- **Sentiment Analysis**:
  - **Recent Product Launch**: Positive reviews from the product launch can fuel further interest in the company, potentially leading to increased sales and revenue growth.
  - **Strong Q3 Earnings**: Reporting strong earnings is a fundamental positive sign, which can contribute to investor confidence and price appreciation.

- **Financial Ratios**:
  - **Profit Margin: 15%**: Indicates healthy profitability, which showcases operational efficiency and the ability to manage costs effectively.
  - **Debt to Equity: 0.3**: A low debt-to-equity ratio suggests a solid balance sheet and lower financial risk.

### Recommendation:

- **Action**: **Consider Buying**
  - Given the current upward price trend, bullish moving averages, and stable RSI, this stock appears to be positioned for future gains.        
  - Positive sentiment from recent product launches and strong earnings further supports a bullish outlook.

- **Target**: Monitor the stock for potential breakout opportunities, especially if it maintains momentum following positive news trends. Re-evaluate the position if RSI approaches overbought levels (70+).

- **Risk Management**: Keep an eye on market conditions and any shifts in sentiment that could affect the stock, including upcoming earnings reports or macroeconomic changes.

------------------------------------------------------------
fundamental_analyst:
### Financial Analysis of TechCorp Inc.

**Key Financial Metrics:**
- **Revenue:** $10 billion
- **Profit Margin:** 15% (indicating efficient cost management and strong profitability)
- **Debt-to-Equity Ratio:** 0.3 (suggests lower financial risk, as the company is less reliant on debt for financing its operations)

**Market Performance:**
- **Price Data:**
  - **Upward Trend** (indicating potential investor confidence)
  - **Relative Strength Index (RSI):** 45 (neutral indicator; suggests no immediate overbought or oversold conditions)
  - **Moving Averages:** Bullish (indicating a positive trend in stock price)

**News Sentiment:**
- **Recent Product Launch:** Received positive reviews (indicates potential for increased revenue and market share)
- **Strong Q3 Earnings:** Suggests stable financial performance and positive growth prospects

### Findings:
- Strong revenue base of $10 billion with a solid profit margin of 15% implies good operational efficiency.
- A low debt-to-equity ratio of 0.3 is a positive indicator, suggesting that TechCorp is well-positioned with minimal leverage.
- The stock is exhibiting an upward price trend with bullish moving averages, which may indicate investor confidence in the stock's appreciation potential.
- Neutral RSI indicates there's still room for potential price movements without being considered overbought or oversold.
- Positive sentiment from recent product launches and strong quarterly earnings suggests increased market demand and potential for future revenue growth.

### Recommendation:
**BUY** - Based on the positive revenue growth potential, strong earnings, and favorable market sentiment, TechCorp Inc. presents a compelling investment opportunity at this time. The company seems to be on a growth trajectory with manageable debt levels and effective cost management. 
"""
