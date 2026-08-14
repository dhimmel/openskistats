import type { SkiAreaSummary } from "./types";

interface NumericRange {
  lower: number;
  lowerInclusive: boolean;
  upper: number;
  upperInclusive: boolean;
}

const NUMBER_PATTERN = "-?(?:\\d+(?:\\.\\d*)?|\\.\\d+)";
const SINGLE_NUMBER = new RegExp(`^(${NUMBER_PATTERN})$`);
const EXPLICIT_RANGE = new RegExp(
  `^([\\[(])\\s*(${NUMBER_PATTERN})?\\s*,\\s*(${NUMBER_PATTERN})?\\s*([\\])])$`,
);

export const INITIAL_COLUMN_FILTERS = [
  { id: "run_count", value: "3" },
  { id: "combined_vertical", value: "50" },
] as const;

function parseNumericRange(filterValue: string): NumericRange | null {
  const expression = filterValue.trim();
  if (expression === "-" || expression === "-0") {
    return {
      lower: Number.NEGATIVE_INFINITY,
      lowerInclusive: true,
      upper: 0,
      upperInclusive: true,
    };
  }

  const singleMatch = expression.match(SINGLE_NUMBER);
  if (singleMatch) {
    const threshold = Number(singleMatch[1]);
    return threshold >= 0
      ? {
          lower: threshold,
          lowerInclusive: true,
          upper: Number.POSITIVE_INFINITY,
          upperInclusive: true,
        }
      : {
          lower: Number.NEGATIVE_INFINITY,
          lowerInclusive: true,
          upper: threshold,
          upperInclusive: true,
        };
  }

  const rangeMatch = expression.match(EXPLICIT_RANGE);
  if (!rangeMatch || (rangeMatch[2] === undefined && rangeMatch[3] === undefined)) {
    return null;
  }
  return {
    lower:
      rangeMatch[2] === undefined
        ? Number.NEGATIVE_INFINITY
        : Number(rangeMatch[2]),
    lowerInclusive: rangeMatch[1] === "[",
    upper:
      rangeMatch[3] === undefined
        ? Number.POSITIVE_INFINITY
        : Number(rangeMatch[3]),
    upperInclusive: rangeMatch[4] === "]",
  };
}

export function matchesNumericFilter(
  value: number | null | undefined,
  filterValue: unknown,
): boolean {
  if (typeof filterValue !== "string" || filterValue.trim() === "") {
    return true;
  }
  const range = parseNumericRange(filterValue);
  if (range === null) {
    return true;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return false;
  }
  const aboveLower = range.lowerInclusive
    ? value >= range.lower
    : value > range.lower;
  const belowUpper = range.upperInclusive
    ? value <= range.upper
    : value < range.upper;
  return aboveLower && belowUpper;
}

export function matchesPercentFilter(
  value: number | null | undefined,
  filterValue: unknown,
): boolean {
  return matchesNumericFilter(value === null || value === undefined ? value : value * 100, filterValue);
}

export function countryCodeToFlag(countryCode: string | null): string | null {
  const normalized = countryCode?.trim().toUpperCase();
  if (!normalized || !/^[A-Z]{2}$/.test(normalized)) {
    return null;
  }
  return String.fromCodePoint(
    ...[...normalized].map((letter) => 0x1f1e6 + letter.charCodeAt(0) - 65),
  );
}

export function matchesCountryFilter(
  skiArea: Pick<SkiAreaSummary, "country" | "country_code">,
  filterValue: unknown,
): boolean {
  if (typeof filterValue !== "string" || filterValue.trim() === "") {
    return true;
  }
  const query = filterValue.trim();
  const queryLower = query.toLocaleLowerCase();
  return (
    skiArea.country?.toLocaleLowerCase().includes(queryLower) === true ||
    skiArea.country_code?.toLocaleLowerCase() === queryLower ||
    countryCodeToFlag(skiArea.country_code) === query
  );
}

export function matchesLatitudeFilter(
  latitude: number | null,
  filterValue: unknown,
): boolean {
  if (typeof filterValue !== "string" || filterValue.trim() === "") {
    return true;
  }
  const query = filterValue.trim();
  if (!/^[a-z]+$/i.test(query)) {
    return matchesNumericFilter(latitude, query);
  }
  if (latitude === null) {
    return false;
  }
  const hemisphere = latitude > 0 ? "north" : latitude < 0 ? "south" : "";
  return hemisphere.includes(query.toLocaleLowerCase());
}

export function meetsInitialInclusionFilters(
  skiArea: Pick<SkiAreaSummary, "run_count" | "combined_vertical">,
): boolean {
  return (
    matchesNumericFilter(skiArea.run_count, INITIAL_COLUMN_FILTERS[0].value) &&
    matchesNumericFilter(
      skiArea.combined_vertical,
      INITIAL_COLUMN_FILTERS[1].value,
    )
  );
}

/**
 * Match when a value is one of the facet values selected in a value picker.
 *
 * `null` and `undefined` are treated alike so that a column's blank option
 * matches rows regardless of how the absence is represented.
 */
export function matchesSetFilter(value: unknown, filterValue: unknown): boolean {
  if (!Array.isArray(filterValue) || filterValue.length === 0) {
    return true;
  }
  const target = value === undefined ? null : value;
  return filterValue.some((entry) => (entry === undefined ? null : entry) === target);
}
