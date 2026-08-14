import { describe, expect, it } from "vitest";

import {
  countryCodeToFlag,
  matchesLatitudeFilter,
  matchesNumericFilter,
  matchesPercentFilter,
  matchesSetFilter,
  searchKey,
} from "../src/filters";

describe("matchesNumericFilter", () => {
  it.each([
    { value: 15, filter: "15", expected: true, purpose: "positive threshold" },
    { value: 14, filter: "15", expected: false, purpose: "below positive threshold" },
    { value: -20, filter: "-20", expected: true, purpose: "negative threshold" },
    { value: -19, filter: "-20", expected: false, purpose: "above negative threshold" },
    { value: 10, filter: "[10, 20]", expected: true, purpose: "inclusive range" },
    { value: 10, filter: "(10, 20)", expected: false, purpose: "exclusive range" },
    { value: 5, filter: "(, 5]", expected: true, purpose: "open lower bound" },
    { value: 20, filter: "[10, )", expected: true, purpose: "open upper bound" },
    { value: 1.5, filter: "1.5", expected: true, purpose: "decimal threshold" },
    { value: 0, filter: "-", expected: true, purpose: "zero shorthand" },
    { value: null, filter: "15", expected: false, purpose: "missing value" },
    { value: null, filter: "", expected: true, purpose: "empty filter" },
    { value: 1, filter: "nonsense", expected: true, purpose: "invalid filter" },
  ])("handles $purpose", ({ value, filter, expected }) => {
    expect(matchesNumericFilter(value, filter)).toBe(expected);
  });
});

it("matches percent filters against displayed values", () => {
  expect(matchesPercentFilter(0.8, "80")).toBe(true);
  expect(matchesPercentFilter(0.79, "80")).toBe(false);
  expect(matchesPercentFilter(null, "80")).toBe(false);
});

it("creates flags only for valid country codes", () => {
  expect(countryCodeToFlag("us")).toBe("🇺🇸");
  expect(countryCodeToFlag(null)).toBeNull();
  expect(countryCodeToFlag("USA")).toBeNull();
});

describe("latitude filters", () => {
  it.each([
    { latitude: 45, filter: "north", expected: true },
    { latitude: 45, filter: "sou", expected: false },
    { latitude: -20, filter: "south", expected: true },
    { latitude: -20, filter: "-20", expected: true },
    { latitude: null, filter: "north", expected: false },
  ])("matches $latitude against $filter", ({ latitude, filter, expected }) => {
    expect(matchesLatitudeFilter(latitude, filter)).toBe(expected);
  });
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
