/** Presentation helpers shared by the ski-area and lift tables. */
import type { CellContext, Column, HeaderContext } from "@tanstack/react-table";
import {
  type CSSProperties,
  type ChangeEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { countryCodeToFlag } from "./filters";
import { formatNumber, MISSING_VALUE } from "./formatters";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends unknown, TValue> {
    cellStyle?: (value: unknown) => CSSProperties;
    className?: string;
    /** Header control to render: a text box by default, or a value picker. */
    filterVariant?: "text" | "faceted";
    filterPlaceholder?: string;
  }

  // Each table supplies its own aggregate shape and narrows it when reading.
  interface TableMeta<TData extends unknown> {
    aggregates: unknown;
  }
}

export function HeaderLabel({
  description,
  focusable = true,
  label,
}: {
  description?: string;
  focusable?: boolean;
  label: ReactNode;
}) {
  if (!description) {
    return label;
  }
  return (
    <span className="oss-table-tooltip-trigger" tabIndex={focusable ? 0 : undefined}>
      {label}
      <span className="oss-table-tooltip" role="tooltip">
        {description}
      </span>
    </span>
  );
}

export function header<TData>(
  label: ReactNode,
  description?: string,
): (context: HeaderContext<TData, unknown>) => ReactNode {
  return ({ column }) => (
    <button
      aria-label={`Sort by ${column.columnDef.id ?? "column"}`}
      className="oss-table-sort-button"
      disabled={!column.getCanSort()}
      onClick={column.getToggleSortingHandler()}
      type="button"
    >
      <HeaderLabel description={description} focusable={false} label={label} />
      {{ asc: "▲", desc: "▼" }[column.getIsSorted() as string] ?? null}
    </button>
  );
}

export function DebouncedInput({
  ariaLabel,
  onChange,
  placeholder,
  value: externalValue,
}: {
  ariaLabel: string;
  onChange: (value: string) => void;
  placeholder?: string;
  value: unknown;
}) {
  const [value, setValue] = useState(String(externalValue ?? ""));

  useEffect(() => {
    setValue(String(externalValue ?? ""));
  }, [externalValue]);

  useEffect(() => {
    const timeout = window.setTimeout(() => onChange(value), 200);
    return () => window.clearTimeout(timeout);
  }, [onChange, value]);

  return (
    <input
      aria-label={ariaLabel}
      className="oss-table-filter"
      onChange={(event: ChangeEvent<HTMLInputElement>) => setValue(event.target.value)}
      placeholder={placeholder ?? "Filter…"}
      type="text"
      value={value}
    />
  );
}

export function textCell(value: string | null): ReactNode {
  return value ?? MISSING_VALUE;
}

/**
 * Render a footer aggregate as a muted label above its value.
 * Stacking keeps narrow metric columns legible without widening the summary row.
 */
export function footerStat(label: string, value: string): ReactNode {
  return (
    <span className="oss-table-footer-stat">
      <span className="oss-table-footer-label">{label}</span>
      <span className="oss-table-footer-value">{value}</span>
    </span>
  );
}

/** Row shape required to render the country cell. */
export interface CountryFields {
  country: string | null;
  country_code: string | null;
}

export function CountryCell<TData extends CountryFields>({
  row,
}: CellContext<TData, unknown>) {
  const country = row.original.country;
  const flag = countryCodeToFlag(row.original.country_code);
  if (country === null && flag === null) {
    return MISSING_VALUE;
  }
  return (
    <span className="oss-table-country">
      {flag && <span aria-hidden="true">{flag}</span>}
      {country && <span>{country}</span>}
    </span>
  );
}

export function LatitudeCell<TData>({ getValue }: CellContext<TData, unknown>) {
  const latitude = getValue<number | null>();
  if (latitude === null) {
    return MISSING_VALUE;
  }
  const intensity = Math.round((Math.abs(latitude) / 90) * 255);
  const background = `rgb(${255 - intensity}, ${255 - intensity}, ${255 - intensity})`;
  return (
    <span
      aria-label={`${Math.abs(latitude).toFixed(1)} degrees ${latitude >= 0 ? "north" : "south"}`}
      className="oss-table-latitude"
      style={{ "--oss-latitude-background": background } as CSSProperties}
    >
      <span aria-hidden="true" className="oss-table-hemisphere">
        {latitude >= 0 ? "ℕ" : "𝕊"}
      </span>
      <span>{Math.abs(latitude).toFixed(1)}°</span>
    </span>
  );
}

export function interpolateColor(
  start: [number, number, number],
  end: [number, number, number],
  value: number,
) {
  const amount = Math.min(1, Math.max(0, value));
  return `rgb(${start.map((channel, index) => Math.round(channel + (end[index] - channel) * amount)).join(", ")})`;
}

export function sequentialColor(value: number) {
  return interpolateColor([255, 255, 255], [161, 0, 191], value);
}

/** Underline a numeric cell proportionally to its share of the column maximum. */
export function metricCell<TData>(
  maximum: number,
  formatter: (value: number | null) => string = formatNumber,
) {
  return ({ getValue }: CellContext<TData, unknown>) => {
    const value = getValue<number | null>();
    const style =
      value === null || maximum <= 0
        ? undefined
        : ({ "--oss-metric-color": sequentialColor(value / maximum) } as CSSProperties);
    return (
      <span className="oss-table-metric" style={style}>
        {formatter(value)}
      </span>
    );
  };
}

/** Largest numeric value of `field`, used to scale metric underlines. */
export function columnMaximum<TData>(
  data: readonly TData[],
  field: keyof TData,
): number {
  return Math.max(
    0,
    ...data.flatMap((row) => (typeof row[field] === "number" ? [row[field]] : [])),
  );
}

/** Options rendered at once before asking the visitor to narrow the search. */
const MAX_FACET_OPTIONS = 200;

/** Label a facet value, including the booleans and blanks columns may hold. */
export function facetOptionLabel(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "(unknown)";
  }
  if (value === true) {
    return "Yes";
  }
  if (value === false) {
    return "No";
  }
  return String(value);
}

/**
 * Multi-select filter listing a column's distinct values with their counts.
 *
 * TanStack computes the counts from rows passing every *other* column's
 * filters, so the options always describe what is still reachable.
 *
 * The popover is positioned as fixed rather than nested in the scroll
 * container, which would otherwise clip it and widen the table's scrollable
 * area.
 */
export function FacetedFilter<TData>({
  ariaLabel,
  column,
}: {
  ariaLabel: string;
  column: Column<TData, unknown>;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [anchor, setAnchor] = useState<{ left: number; top: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const selected = (column.getFilterValue() as unknown[] | undefined) ?? [];
  const selectedSet = new Set(selected);
  const facets = column.getFacetedUniqueValues();

  // Most frequent first, since that is the useful end of a long tail.
  const options = useMemo(
    () =>
      [...facets.entries()]
        .map(([value, count]) => ({ count, label: facetOptionLabel(value), value }))
        .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label)),
    [facets],
  );
  const needle = query.trim().toLocaleLowerCase();
  const matches = needle
    ? options.filter((option) => option.label.toLocaleLowerCase().includes(needle))
    : options;
  const shown = matches.slice(0, MAX_FACET_OPTIONS);

  const reposition = () => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) {
      const width = 260;
      setAnchor({
        left: Math.min(Math.max(8, rect.left), window.innerWidth - width - 8),
        top: rect.bottom + 4,
      });
    }
  };

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (
        !popoverRef.current?.contains(target) &&
        !triggerRef.current?.contains(target)
      ) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    const onReposition = () => reposition();
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onReposition);
    window.addEventListener("scroll", onReposition, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onReposition);
      window.removeEventListener("scroll", onReposition, true);
    };
  }, [open]);

  const setSelection = (values: unknown[]) =>
    column.setFilterValue(values.length === 0 ? undefined : values);

  return (
    <>
      <button
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={ariaLabel}
        className="oss-table-facet-trigger"
        onClick={() => {
          reposition();
          setOpen((value) => !value);
        }}
        ref={triggerRef}
        type="button"
      >
        {selected.length === 0 ? "Any" : `${formatNumber(selected.length)} selected`}
      </button>
      {open && anchor && (
        <div
          className="oss-table-facet-popover"
          ref={popoverRef}
          role="dialog"
          style={{ left: anchor.left, top: anchor.top }}
        >
          <input
            aria-label={`Search ${ariaLabel}`}
            autoFocus
            className="oss-table-facet-search"
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              setQuery(event.target.value)
            }
            placeholder="Search values…"
            type="text"
            value={query}
          />
          <div className="oss-table-facet-actions">
            <button
              disabled={matches.length === 0}
              onClick={() =>
                setSelection([
                  ...new Set([...selected, ...matches.map((option) => option.value)]),
                ])
              }
              type="button"
            >
              Select {formatNumber(matches.length)}
            </button>
            <button
              disabled={selected.length === 0}
              onClick={() => setSelection([])}
              type="button"
            >
              Clear
            </button>
          </div>
          <div className="oss-table-facet-list">
            {shown.length === 0 ? (
              <p className="oss-table-facet-empty">No matching values.</p>
            ) : (
              shown.map(({ count, label, value }) => (
                <label className="oss-table-facet-option" key={label}>
                  <input
                    checked={selectedSet.has(value)}
                    onChange={() =>
                      setSelection(
                        selectedSet.has(value)
                          ? selected.filter((entry) => entry !== value)
                          : [...selected, value],
                      )
                    }
                    type="checkbox"
                  />
                  <span className="oss-table-facet-value">{label}</span>
                  <span className="oss-table-facet-count">{formatNumber(count)}</span>
                </label>
              ))
            )}
          </div>
          {matches.length > shown.length && (
            <p className="oss-table-facet-more">
              Showing {formatNumber(shown.length)} of {formatNumber(matches.length)}.
              Keep typing to narrow.
            </p>
          )}
        </div>
      )}
    </>
  );
}
