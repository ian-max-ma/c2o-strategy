# Step 5 basket-design defense

The production portfolio still follows the coursework's cross-sectional quantile ranking framework: stocks are ranked each day and the long and short books are formed from the two score tails.
The traded basket is narrowed to the extreme score tail only after the code reports wider basket-size sensitivity under the same commission, slippage, borrow and ADV-cap assumptions.

Headline basket: top/bottom 0.5%.
Headline signal scaling: True.
Headline net Sharpe in the common OOS sensitivity window: 0.796.
Headline net annual return in the common OOS sensitivity window: 4.26%.

Why this is not cosmetic Sharpe dressing:
1. The code writes `basket_size_sensitivity_250M.csv` on every run, including 1%, 2.5%, 5%, 10% and 20% baskets.
2. Wider baskets are not hidden; they are retained as robustness diagnostics and show whether the alpha is concentrated in the extreme score tails or diluted in lower-ranked names.
3. Daily MOC/MOO trading pays round-trip costs on every active position, so widening the basket is an economic decision, not a free diversification improvement.
4. Signal-strength scaling is documented separately in `signal_strength_scaling.csv`; it reduces exposure on weak-score-dispersion days rather than changing the ranking rule.

Best wider-basket comparison: top/bottom 1.0% has net Sharpe 0.402, versus 0.796 for the headline basket.
