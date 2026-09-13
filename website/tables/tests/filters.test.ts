import { describe, expect, it } from "vitest";

import {
  countryCodeToFlag,
  isNumericRange,
  matchesNumericFilter,
  matchesPercentFilter,
  matchesSetFilter,
  searchKey,
} from "../src/filters";

const atLeast = (lower: number) => ({ lower, upper: Number.POSITIVE_INFINITY });
const atMost = (upper: number) => ({ lower: Number.NEGATIVE_INFINITY, upper });

describe("matchesNumericFilter", () => {
  it.each([
    { value: 15, filter: atLeast(15), expected: true, purpose: "a lower bound, inclusive" },
    { value: 14, filter: atLeast(15), expected: false, purpose: "below the lower bound" },
    { value: -20, filter: atMost(-20), expected: true, purpose: "a negative upper bound, inclusive" },
    { value: -19, filter: atMost(-20), expected: false, purpose: "above the upper bound" },
    { value: 10, filter: { lower: 10, upper: 20 }, expected: true, purpose: "the lower edge of a closed range" },
    { value: 20, filter: { lower: 10, upper: 20 }, expected: true, purpose: "the upper edge of a closed range" },
    { value: 20.5, filter: { lower: 10, upper: 20 }, expected: false, purpose: "past a closed range" },
    { value: 1.5, filter: atLeast(1.5), expected: true, purpose: "a decimal bound" },
    { value: null, filter: atLeast(15), expected: false, purpose: "a missing value" },
    { value: null, filter: undefined, expected: true, purpose: "no filter" },
    { value: 1, filter: "3", expected: true, purpose: "a filter that is not a range" },
  ])("handles $purpose", ({ value, filter, expected }) => {
    expect(matchesNumericFilter(value, filter)).toBe(expected);
  });
});

it("recognises only objects with two numeric bounds as ranges", () => {
  expect(isNumericRange({ lower: 1, upper: 2 })).toBe(true);
  expect(isNumericRange(atLeast(1))).toBe(true);
  expect(isNumericRange({ lower: 1 })).toBe(false);
  expect(isNumericRange(["a"])).toBe(false);
  expect(isNumericRange(null)).toBe(false);
});

it("matches percent filters against displayed values", () => {
  expect(matchesPercentFilter(0.8, atLeast(80))).toBe(true);
  expect(matchesPercentFilter(0.79, atLeast(80))).toBe(false);
  expect(matchesPercentFilter(null, atLeast(80))).toBe(false);
});

it("creates flags only for valid country codes", () => {
  expect(countryCodeToFlag("us")).toBe("🇺🇸");
  expect(countryCodeToFlag(null)).toBeNull();
  expect(countryCodeToFlag("USA")).toBeNull();
});

describe("matchesSetFilter", () => {
  it.each([
    { value: "gondola", filter: undefined, expected: true, purpose: "no selection keeps every row" },
    { value: "gondola", filter: [], expected: true, purpose: "empty selection keeps every row" },
    { value: "gondola", filter: ["gondola"], expected: true, purpose: "selected value" },
    { value: "gondola", filter: ["chair_lift"], expected: false, purpose: "unselected value" },
    { value: "gondola", filter: ["chair_lift", "gondola"], expected: true, purpose: "one of several selected" },
    { value: true, filter: [true], expected: true, purpose: "boolean value" },
    { value: false, filter: [true], expected: false, purpose: "opposite boolean" },
    { value: null, filter: [null], expected: true, purpose: "blank option" },
    { value: undefined, filter: [null], expected: true, purpose: "undefined matches the blank option" },
    { value: "gondola", filter: [null], expected: false, purpose: "blank option excludes present values" },
  ])("$purpose", ({ value, filter, expected }) => {
    expect(matchesSetFilter(value, filter)).toBe(expected);
  });
});

describe("searchKey", () => {
  it.each([
    { query: "chairlift", option: "chair_lift", purpose: "a missing separator" },
    { query: "chair lift", option: "chair_lift", purpose: "a space for an underscore" },
    { query: "t bar", option: "t-bar", purpose: "a space for a hyphen" },
    { query: "val d isere", option: "Val d'Isère", purpose: "dropped accents" },
    { query: "US", option: "US", purpose: "a country code" },
  ])("folds $purpose", ({ query, option }) => {
    expect(searchKey(option).includes(searchKey(query))).toBe(true);
  });

  it("keeps a flag emoji searchable", () => {
    expect(searchKey("🇫🇷")).toBe("🇫🇷");
  });

  it("does not collapse distinct values", () => {
    expect(searchKey("gondola").includes(searchKey("funicular"))).toBe(false);
  });
});
