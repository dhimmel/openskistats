/** Presentation helpers shared by the ski-area and lift tables. */
import type { CellContext, HeaderContext } from "@tanstack/react-table";
import {
  type CSSProperties,
  type ChangeEvent,
  type ReactNode,
  useEffect,
  useState,
} from "react";

import { countryCodeToFlag } from "./filters";
import { formatNumber, MISSING_VALUE } from "./formatters";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends unknown, TValue> {
    cellStyle?: (value: unknown) => CSSProperties;
    className?: string;
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
