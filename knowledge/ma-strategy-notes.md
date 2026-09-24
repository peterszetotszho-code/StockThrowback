# Moving Average Crossover Strategies — Notes

## How they work

A dual moving-average crossover buys when a fast MA crosses above a slow MA
(golden cross) and sells on the reverse (death cross). It captures sustained
trends at the cost of lagging the price.

## Known failure modes

1. **Whipsaw in ranging markets.** When price oscillates sideways, the fast and
   slow MAs repeatedly cross, generating many false signals and small losses.
2. **Late exits in reversals.** Because MAs lag, a strategy often exits after a
   trend has already reversed, giving back a large share of its gains.
3. **Drawdown concentration.** Trend followers earn most of their return from a
   few large trends; deep drawdowns are common between them.

## Mitigations

- Add a trend filter (e.g. MACD direction) or a volatility filter to avoid
  trading in choppy regimes.
- Combine with a regime label (bullish / bearish / ranging) to skip ranging
  periods entirely.
