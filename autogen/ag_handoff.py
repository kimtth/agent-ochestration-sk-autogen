import os
from dataclasses import dataclass
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    default_subscription,
)
from autogen_core.models import SystemMessage, UserMessage, ChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentRequest:
    company: str
    inquiry: str
    request_id: str


@dataclass
class HandoffResponse:
    handler: str
    content: str
    request_id: str


@default_subscription
class TriageAdvisor(RoutedAgent):
    def __init__(self, client):
        super().__init__("Triage Advisor")
        self._system = SystemMessage(
            content="Route refund inquiries to 'refund' and investment questions to 'specialist'."
        )
        self._client = client

    @message_handler
    async def on_request(self, message: InvestmentRequest, ctx: MessageContext) -> None:
        result = await self._client.create(
            messages=[
                self._system,
                UserMessage(content=message.inquiry, source="TriageAdvisor"),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        choice = "specialist" if "invest" in message.inquiry.lower() else "refund"
        await self.publish_message(
            HandoffResponse(
                handler=choice, content=result.content, request_id=message.request_id
            ),
            topic_id=DefaultTopicId(),
        )


@default_subscription
class EquitySpecialist(RoutedAgent):
    def __init__(self, client):
        super().__init__("Equity Specialist")
        self._system = SystemMessage(content="Provide detailed investment analysis.")
        self._client = client

    @message_handler
    async def on_handoff(self, message: HandoffResponse, ctx: MessageContext) -> None:
        if message.handler != "specialist":
            return
        result = await self._client.create(
            messages=[
                self._system,
                UserMessage(content=message.content, source="EquitySpecialist"),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        print(f"Equity Specialist report ({message.request_id}):\n{result.content}")


@default_subscription
class HumanAdvisor(RoutedAgent):
    def __init__(self, client):
        super().__init__("Human Advisor")
        self._system = SystemMessage(
            content="Assist with refund and complex inquiries."
        )
        self._client = client

    @message_handler
    async def on_handoff(self, message: HandoffResponse, ctx: MessageContext) -> None:
        if message.handler != "refund":
            return
        print(
            f"Human Advisor handling refund ({message.request_id}):\n{message.content}"
        )


async def run_handoff_scenario():
    runtime = SingleThreadedAgentRuntime()
    client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="gpt-4o-mini",  # a name of OpenAI model (not clear meaning of this value when using Azure OpenAI)
    )
    await TriageAdvisor.register(runtime, "triage", lambda: TriageAdvisor(client))
    await EquitySpecialist.register(
        runtime, "specialist", lambda: EquitySpecialist(client)
    )
    await HumanAdvisor.register(runtime, "human", lambda: HumanAdvisor(client))

    runtime.start()
    await runtime.publish_message(
        InvestmentRequest(
            company="TechCorp",
            inquiry="I want to invest in stocks",
            request_id="HR-001",
        ),
        topic_id=DefaultTopicId(),
    )
    await runtime.stop_when_idle()
    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_handoff_scenario())

"""
Equity Specialist report (HR-001):
Certainly! When conducting an investment analysis, it's important to consider several key factors. Below is a structured approach for evaluating potential equity investments:

### 1. **Company Overview**
   - **Business Model**: Understand how the company generates revenue. Is it a product-based, service-oriented, or subscription model?
   - **Market Position**: Identify the company's position in its industry. Is it a market leader, challenger, or niche player?
   - **Management Team**: Assess the experience and track record of the leadership. Effective management is critical for executing strategies and navigating challenges.

### 2. **Financial Performance**
   - **Income Statement Analysis**:
     - Revenue Growth: Analyze year-over-year revenue growth to assess demand and market share expansion.
     - Profit Margins: Look at gross, operating, and net profit margins to evaluate efficiency.
   - **Balance Sheet Analysis**:
     - Debt Levels: Examine total debt relative to equity (Debt-to-Equity ratio) and EBIT coverage.
     - Liquidity Ratios: Assess current and quick ratios to determine short-term financial health.
   - **Cash Flow Statement**: Focus on operating cash flow to ensure the company generates sufficient cash from its core operations.

### 3. **Valuation Metrics**
   - **Price-to-Earnings (P/E) Ratio**: Compare the company's P/E ratio with industry averages to gauge valuation.
   - **Price-to-Book (P/B) Ratio**: Analyze P/B versus competitors, particularly important for financial companies.
   - **Enterprise Value (EV) to EBITDA**: A useful metric for understanding overall company valuation.      
   - **Discounted Cash Flow (DCF) Analysis**: Project future cash flows and discount them back to present value for intrinsic valuation.

### 4. **Market Trends and Competitive Landscape**
   - **Industry Trends**: Identify macroeconomic factors, technological advancements, and regulatory changes affecting the industry.
   - **Competitive Analysis**: Assess competitors’ strengths and weaknesses, market share, and barriers to entry.
   - **SWOT Analysis**: Evaluate the company’s strengths, weaknesses, opportunities, and threats.

### 5. **Investment Outlook**
   - **Growth Potential**: Analyze potential for future growth in revenue and market expansion.
   - **Risks**: Identify key risks (market risk, operational risk, regulatory risk, etc.) and how they can impact the investment.
   - **Dividend Policy**: Consider whether the company pays dividends and its history of dividend growth as an aspect of total return.

### 6. **Technical Analysis (if applicable)**
   - **Price Trends**: Analyze historical price movements and trends to gain insights on entry and exit points.
   - **Volume Patterns**: Look for abnormal trading volumes that might signal investor interest.

### 7. **Investment Thesis Development**
   - **Construct a clear thesis** based on your analysis indicating why the investment is attractive, the expected return, and the timeframe.
   - **Set Price Targets**: Determine realistic price targets based on your valuation and analysis.
   - **Risk Management**: Establish stop-loss levels or other risk mitigation strategies.

### Conclusion
An investment analysis is not just about financial metrics; it requires a comprehensive understanding of various qualitative and quantitative aspects. The blend of detailed financial scrutiny, awareness of market dynamics, and a sound investment thesis are critical for making informed decisions.

If you have a specific stock or sector in mind, please provide that information for a more focused analysis!
"""
