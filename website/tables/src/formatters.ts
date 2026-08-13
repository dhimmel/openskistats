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

export function formatMeters(value: number | null): string {
  const number = formatNumber(value);
  return number === MISSING_VALUE ? number : `${number}${NARROW_NO_BREAK_SPACE}m`;
}

export function formatPercent(value: number | null): string {
  return value === null ? MISSING_VALUE : `${formatNumber(value * 100)}%`;
}
