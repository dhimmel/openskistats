/**
 * A closed interval, inclusive at both ends. Either endpoint may be infinite.
 *
 * The range popovers hand this object to the filter function directly, so a
 * typed bound and a dragged bound are the same piece of state.
 */
export interface NumericRange {
  lower: number;
  upper: number;
}

export const INITIAL_COLUMN_FILTERS = [
  { id: "run_count", value: { lower: 3, upper: Number.POSITIVE_INFINITY } },
  { id: "combined_vertical", value: { lower: 50, upper: Number.POSITIVE_INFINITY } },
] as const;

export function isNumericRange(value: unknown): value is NumericRange {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as NumericRange).lower === "number" &&
    typeof (value as NumericRange).upper === "number"
  );
}

/** Whether `value` falls inside the interval. */
export function rangeContains(range: NumericRange, value: number): boolean {
  return value >= range.lower && value <= range.upper;
}

export function matchesNumericFilter(
  value: number | null | undefined,
  filterValue: unknown,
): boolean {
  if (!isNumericRange(filterValue)) {
    return true;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return false;
  }
  return rangeContains(filterValue, value);
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
