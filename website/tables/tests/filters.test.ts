import { describe, expect, it } from "vitest";

import {
  countryCodeToFlag,
  matchesCountryFilter,
  matchesLatitudeFilter,
  matchesNumericFilter,
  matchesPercentFilter,
  meetsInitialInclusionFilters,
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

describe("country filters", () => {
  const france = { country: "France", country_code: "FR" };

  it.each(["fran", "FR", "fr", "🇫🇷"])("matches %s", (filter) => {
    expect(matchesCountryFilter(france, filter)).toBe(true);
  });
  it("does not match another country", () => {
    expect(matchesCountryFilter(france, "US")).toBe(false);
  });
  it("creates flags only for valid country codes", () => {
    expect(countryCodeToFlag("us")).toBe("🇺🇸");
    expect(countryCodeToFlag(null)).toBeNull();
    expect(countryCodeToFlag("USA")).toBeNull();
  });
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

it("applies the initial public table inclusion filters", () => {
  expect(meetsInitialInclusionFilters({ run_count: 3, combined_vertical: 50 })).toBe(
    true,
  );
  expect(meetsInitialInclusionFilters({ run_count: 2, combined_vertical: 500 })).toBe(
    false,
  );
  expect(meetsInitialInclusionFilters({ run_count: 20, combined_vertical: null })).toBe(
    false,
  );
});
