import type { SkiAreaSummary } from "./types";

/**
 * A half-open or closed interval, as written in the `[lower, upper)` grammar
 * the numeric filter boxes accept. Either endpoint may be infinite.
 */
export interface NumericRange {
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

/**
 * Read a filter box's text as an interval, or `null` when it is not one.
 *
 * The range popovers parse the filter they are about to replace so that a
 * typed bound and a dragged bound are the same piece of state.
 */
export function parseNumericRange(filterValue: string): NumericRange | null {
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

/** Whether `value` falls inside the interval, honouring each end's bracket. */
export function rangeContains(range: NumericRange, value: number): boolean {
  const aboveLower = range.lowerInclusive
    ? value >= range.lower
    : value > range.lower;
  const belowUpper = range.upperInclusive
    ? value <= range.upper
    : value < range.upper;
  return aboveLower && belowUpper;
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
  return rangeContains(range, value);
}

export function matchesPercentFilter(
  value: number | null | undefined,
  filterValue: unknown,
): boolean {
  return matchesNumericFilter(value === null || value === undefined ? value : value * 100, filterValue);
}

/**
 * Fold text to the form a value picker searches on.
 *
 * Case, accents, and every separator are dropped, so `chairlift`, `chair lift`,
 * and `chair_lift` all reduce alike, as do `Val d'Isere` and `Val d'Isère`.
 * Symbols survive, which is what lets a flag emoji stay searchable.
 *
 * Combining marks are stripped wholesale, so Japanese dakuten fold away too
 * and `か` matches `が`. That widens a search rather than misdirecting it.
 */
export function searchKey(text: string): string {
  return text
    .normalize("NFD")
    .replace(/\p{M}+/gu, "")
    .toLocaleLowerCase()
    .replace(/[\p{P}\p{Z}\p{C}]+/gu, "");
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

/**
 * Selections are looked up once per filtered row, so a large one is indexed
 * rather than scanned. The array identity is stable while the filter is
 * unchanged, which makes it a sound cache key.
 */
const selectionSets = new WeakMap<object, Set<unknown>>();

function selectionSet(filterValue: unknown[]): Set<unknown> {
  let set = selectionSets.get(filterValue);
  if (set === undefined) {
    set = new Set(filterValue.map((entry) => entry ?? null));
    selectionSets.set(filterValue, set);
  }
  return set;
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
  return selectionSet(filterValue).has(value ?? null);
}
