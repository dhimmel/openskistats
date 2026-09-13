import { expect, it } from "vitest";

import {
  formatLatitude,
  formatLongitude,
  formatMeters,
  formatNumber,
  formatPercent,
  MISSING_VALUE,
} from "../src/formatters";

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

it.each([
  { value: 45.25, expected: "45.3°N", purpose: "a northern latitude" },
  { value: -12.5, expected: "12.5°S", purpose: "a southern latitude" },
  { value: 0, expected: "0°", purpose: "the equator, which has no hemisphere" },
])("names $purpose", ({ value, expected }) => {
  expect(formatLatitude(value)).toBe(expected);
});

it.each([
  { value: 140.25, expected: "140.3°E", purpose: "an eastern longitude" },
  { value: -72.5, expected: "72.5°W", purpose: "a western longitude" },
  { value: 0, expected: "0°", purpose: "the prime meridian" },
  { value: -180, expected: "180°W", purpose: "the western limit" },
  { value: 180, expected: "180°E", purpose: "the eastern limit" },
])("names $purpose", ({ value, expected }) => {
  expect(formatLongitude(value)).toBe(expected);
});
