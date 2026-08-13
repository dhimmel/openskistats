import { expect, it } from "vitest";

import { formatMeters, formatNumber, formatPercent, MISSING_VALUE } from "../src/formatters";

it("formats numbers while preserving missing values", () => {
  expect(formatNumber(1234.56)).toBe((1235).toLocaleString());
  expect(formatNumber(2.338, 1)).toBe((2.3).toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }));
  expect(formatNumber(null)).toBe(MISSING_VALUE);
});

it("formats meters and percentages", () => {
  expect(formatMeters(1234)).toBe(`${(1234).toLocaleString()}\u202fm`);
  expect(formatMeters(null)).toBe(MISSING_VALUE);
  expect(formatPercent(0.761)).toBe("76%");
  expect(formatPercent(null)).toBe(MISSING_VALUE);
});
