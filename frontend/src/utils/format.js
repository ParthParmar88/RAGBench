// Display formatting helpers for token/cost metadata.

const tokenFormatter = new Intl.NumberFormat("en-US");

export function formatTokens(n) {
  return tokenFormatter.format(n ?? 0);
}

// Small costs need more precision (e.g. $0.0018); larger ones read fine at 2dp.
export function formatCost(cost, currency = "USD") {
  if (cost == null) return "—";
  const digits = cost < 1 ? 4 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(cost);
}
