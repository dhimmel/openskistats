const NARROW_NO_BREAK_SPACE = "\u202f";
export const MISSING_VALUE = "—";

export function formatNumber(value: number | null, digits = 0): string {
  if (value === null || !Number.isFinite(value)) {
    return MISSING_VALUE;
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/**
 * Format a filter bound with no more decimals than it needs.
 *
 * Bounds come from bin edges, whose width varies by column, so a fixed number
 * of decimals would print either `3.0` runs or a truncated `0.2` percent.
 */
export function formatBound(value: number, digits = 2): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function formatMeters(value: number | null): string {
  const number = formatNumber(value);
  return number === MISSING_VALUE ? number : `${number}${NARROW_NO_BREAK_SPACE}m`;
}

export function formatPercent(value: number | null): string {
  return value === null ? MISSING_VALUE : `${formatNumber(value * 100)}%`;
}
