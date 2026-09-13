/**
 * Shareable table state in the query string.
 *
 * Filters and sorting are read from `location.search` and written back with
 * `history.replaceState`, so copying the address bar captures the current
 * view. Only state that differs from a table's defaults is written, and a key
 * with an empty value, `run_count=`, clears a default filter.
 *
 * Numeric columns write an inclusive range with either side optional, as
 * GitHub search does: `run_count=3..`, `latitude=..0`, `latitude=-45..-30`.
 * Value pickers repeat the key once per selected value in sorted order,
 * `country=AT&country=CH`, with the blank option written as `null`.
 * Sorting follows JSON:API: `sort=-lift_count` descends, `sort=ski_area_name`
 * ascends, and a comma joins several in priority order.
 *
 * Reading is lenient so that old or hand-edited links degrade gracefully:
 * a parameter that fails to parse is treated as absent, so the default
 * applies, and unknown values, columns, and parameters are ignored.
 */
import {
  type ColumnDef,
  type ColumnFiltersState,
  functionalUpdate,
  type OnChangeFn,
  type RowData,
  type SortingState,
} from "@tanstack/react-table";
import { useCallback, useMemo, useSyncExternalStore } from "react";

import { isNumericRange, type NumericRange } from "./filters";
import { restricts } from "./range";
import type { TableFeatures } from "./table-features";

/** How a value picker's option is written to and read from the query string. */
export interface UrlValueCodec {
  /** The URL form of a facet value. */
  encode: (value: unknown) => string;
  /** The facet value a URL form names, or `undefined` when it names none. */
  decode: (text: string) => unknown;
}

/** What the codec needs to know about a filterable column. */
export interface UrlColumn {
  id: string;
  /** Written as a range, or as one key per selected value. */
  variant: "faceted" | "range";
  /** Facet translation beyond the value's own text, as country uses for its code. */
  urlValue?: UrlValueCodec;
}

export interface TableUrlState {
  columnFilters: ColumnFiltersState;
  sorting: SortingState;
}

export interface TableUrlSpec {
  columns: readonly UrlColumn[];
  /** The state an untouched table shows, which the URL leaves unwritten. */
  defaults: TableUrlState;
}

/** Facet values that are booleans, as the lift table's detachable column holds. */
export const BOOLEAN_URL_VALUE: UrlValueCodec = {
  encode: String,
  decode: (text) => (text === "true" ? true : text === "false" ? false : undefined),
};

const SORT_KEY = "sort";

/** The blank option of a value picker, which matches rows with no value. */
const BLANK = "null";

const RANGE_SEPARATOR = "..";

/**
 * A decimal number as `String` writes one, exponent included, so that every
 * bound the writer produces reads back.
 */
const NUMBER = /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

function parseEnd(text: string, open: number): number | undefined {
  if (text === "") {
    return open;
  }
  if (!NUMBER.test(text)) {
    return undefined;
  }
  const value = Number(text);
  return Number.isFinite(value) ? value : undefined;
}

/** Read `lower..upper`, or `undefined` when the text is not a usable range. */
function parseRange(text: string): NumericRange | undefined {
  const separator = text.indexOf(RANGE_SEPARATOR);
  if (separator === -1) {
    return undefined;
  }
  const lower = parseEnd(text.slice(0, separator), Number.NEGATIVE_INFINITY);
  const upper = parseEnd(
    text.slice(separator + RANGE_SEPARATOR.length),
    Number.POSITIVE_INFINITY,
  );
  if (lower === undefined || upper === undefined || lower > upper) {
    return undefined;
  }
  const range = { lower, upper };
  return restricts(range) ? range : undefined;
}

function formatEnd(value: number): string {
  return Number.isFinite(value) ? String(value) : "";
}

function formatRange(range: NumericRange): string {
  return `${formatEnd(range.lower)}${RANGE_SEPARATOR}${formatEnd(range.upper)}`;
}

function encodeFacet(column: UrlColumn, value: unknown): string {
  if (value === null || value === undefined) {
    return BLANK;
  }
  return column.urlValue?.encode(value) ?? String(value);
}

function decodeFacet(column: UrlColumn, text: string): unknown {
  if (text === BLANK) {
    return null;
  }
  return column.urlValue === undefined ? text : column.urlValue.decode(text);
}

/**
 * A filter value as the query string would carry it, or `undefined` for none.
 *
 * Facet values are deduplicated and sorted so that selection order does not
 * change the link, and so that two selections compare by their written form.
 */
function urlValues(column: UrlColumn, filterValue: unknown): string[] | undefined {
  if (column.variant === "range") {
    return isNumericRange(filterValue) ? [formatRange(filterValue)] : undefined;
  }
  if (!Array.isArray(filterValue) || filterValue.length === 0) {
    return undefined;
  }
  return [...new Set(filterValue.map((value) => encodeFacet(column, value)))].sort();
}

/** A column's filter as the query string states it. */
type UrlFilter = "cleared" | "default" | { value: unknown };

function readColumn(column: UrlColumn, texts: readonly string[]): UrlFilter {
  if (texts.length === 0) {
    return "default";
  }
  // An empty value alone clears the filter; beside real values it is noise.
  const given = texts.filter((text) => text !== "");
  if (given.length === 0) {
    return "cleared";
  }
  if (column.variant === "range") {
    const range = parseRange(given[0]);
    return range === undefined ? "default" : { value: range };
  }
  const values = [
    ...new Set(
      given
        .map((text) => decodeFacet(column, text))
        .filter((value) => value !== undefined),
    ),
  ];
  return values.length === 0 ? "default" : { value: values };
}

function formatSorting(sorting: SortingState): string {
  return sorting.map((sort) => `${sort.desc ? "-" : ""}${sort.id}`).join(",");
}

function readSorting(text: string | null, spec: TableUrlSpec): SortingState {
  if (text === null) {
    return spec.defaults.sorting;
  }
  if (text === "") {
    return [];
  }
  const ids = new Set(spec.columns.map((column) => column.id));
  const sorting = text.split(",").flatMap((entry) => {
    const desc = entry.startsWith("-");
    const id = desc ? entry.slice(1) : entry;
    return ids.has(id) ? [{ desc, id }] : [];
  });
  return sorting.length === 0 ? spec.defaults.sorting : sorting;
}

function filterMap(state: TableUrlState): Map<string, unknown> {
  return new Map(state.columnFilters.map((filter) => [filter.id, filter.value]));
}

/** The table state a query string describes, with defaults where it is silent. */
export function readTableState(search: string, spec: TableUrlSpec): TableUrlState {
  const params = new URLSearchParams(search);
  const defaults = filterMap(spec.defaults);
  const columnFilters: ColumnFiltersState = [];
  for (const column of spec.columns) {
    const read = readColumn(column, params.getAll(column.id));
    const value =
      read === "default"
        ? defaults.get(column.id)
        : read === "cleared"
          ? undefined
          : read.value;
    if (value !== undefined) {
      columnFilters.push({ id: column.id, value });
    }
  }
  return { columnFilters, sorting: readSorting(params.get(SORT_KEY), spec) };
}

function sameValues(a: string[] | undefined, b: string[] | undefined): boolean {
  if (a === undefined || b === undefined) {
    return a === b;
  }
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/**
 * The query string that describes `state`, starting from the current one so
 * that parameters the table does not own survive.
 */
export function writeTableState(
  search: string,
  state: TableUrlState,
  spec: TableUrlSpec,
): string {
  const params = new URLSearchParams(search);
  const current = filterMap(state);
  const defaults = filterMap(spec.defaults);
  for (const column of spec.columns) {
    params.delete(column.id);
    const values = urlValues(column, current.get(column.id));
    if (sameValues(values, urlValues(column, defaults.get(column.id)))) {
      continue;
    }
    for (const value of values ?? [""]) {
      params.append(column.id, value);
    }
  }
  params.delete(SORT_KEY);
  const sorting = formatSorting(state.sorting);
  if (sorting !== formatSorting(spec.defaults.sorting)) {
    params.set(SORT_KEY, sorting);
  }
  const text = params.toString();
  return text === "" ? "" : `?${text}`;
}

/** The URL-visible description of each filterable leaf column. */
export function urlColumnsFrom<TData extends RowData>(
  columns: readonly ColumnDef<TableFeatures, TData, unknown>[],
): UrlColumn[] {
  return columns.flatMap((column) => {
    if ("columns" in column && column.columns !== undefined) {
      return urlColumnsFrom(column.columns);
    }
    const variant = column.meta?.filterVariant;
    if (column.id === undefined || variant === undefined) {
      return [];
    }
    return [{ id: column.id, urlValue: column.meta?.urlValue, variant }];
  });
}

/**
 * The query string as an external store shared by every table on the page.
 *
 * `popstate` matters because Quarto's section anchors push history entries
 * of their own: set Austria, click a heading link, change to Switzerland,
 * press Back, and the URL is on Austria again while a naive hook would still
 * show Switzerland.
 */
const listeners = new Set<() => void>();

export const searchStore = {
  getSnapshot: (): string => window.location.search,
  subscribe: (listener: () => void): (() => void) => {
    listeners.add(listener);
    window.addEventListener("popstate", listener);
    return () => {
      listeners.delete(listener);
      window.removeEventListener("popstate", listener);
    };
  },
};

/**
 * Replace the query string in place, keeping the path, fragment, and history
 * entry, and tell every subscribed table.
 *
 * Callers write only when a change settles, never per drag frame or
 * keystroke, since Safari rate-limits `replaceState`.
 */
export function replaceSearch(search: string): void {
  if (search === searchStore.getSnapshot()) {
    return;
  }
  const { hash, pathname } = window.location;
  window.history.replaceState(window.history.state, "", `${pathname}${search}${hash}`);
  for (const listener of listeners) {
    listener();
  }
}

/**
 * Filters and sorting backed by the query string rather than component state.
 *
 * The URL is the source of truth: setters write it, and the table re-reads
 * it, so a change made by Back or Forward shows just as a click does.
 */
export function useUrlTableState(spec: TableUrlSpec): TableUrlState & {
  setColumnFilters: OnChangeFn<ColumnFiltersState>;
  setSorting: OnChangeFn<SortingState>;
} {
  const search = useSyncExternalStore(searchStore.subscribe, searchStore.getSnapshot);
  const state = useMemo(() => readTableState(search, spec), [search, spec]);
  const update = useCallback(
    (change: (current: TableUrlState) => TableUrlState) => {
      // Read the URL as it stands rather than as this render saw it, since
      // updaters may arrive in sequence before a re-render.
      const search = searchStore.getSnapshot();
      const current = readTableState(search, spec);
      replaceSearch(writeTableState(search, change(current), spec));
    },
    [spec],
  );
  const setColumnFilters = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updater) =>
      update((current) => ({
        ...current,
        columnFilters: functionalUpdate(updater, current.columnFilters),
      })),
    [update],
  );
  const setSorting = useCallback<OnChangeFn<SortingState>>(
    (updater) =>
      update((current) => ({
        ...current,
        sorting: functionalUpdate(updater, current.sorting),
      })),
    [update],
  );
  return { ...state, setColumnFilters, setSorting };
}
