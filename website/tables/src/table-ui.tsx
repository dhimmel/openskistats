/** Presentation helpers shared by the ski-area and lift tables. */
import type {
  CellContext,
  Column,
  HeaderContext,
  Row,
} from "@tanstack/react-table";
import {
  type CSSProperties,
  type ChangeEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  countryCodeToFlag,
  type NumericRange,
  rangeContains,
} from "./filters";
import { formatBound, formatNumber, MISSING_VALUE } from "./formatters";
import {
  boundsFromEdges,
  boundsFromFilter,
  buildHistogram,
  describeBounds,
  formatRangeFilter,
  type Histogram,
  type HistogramBin,
  roundTo,
  UNBOUNDED,
} from "./range";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends unknown, TValue> {
    cellStyle?: (value: unknown) => CSSProperties;
    className?: string;
    /** Order a value picker's options: by descending count, or by label. */
    facetSort?: "count" | "label";
    /**
     * Multiply values by this before binning and filtering, so a column stored
     * as a fraction filters in the whole percent its cells display.
     */
    filterScale?: number;
    /** Format a bound for the range popover's labels and summary. */
    filterFormat?: (value: number) => string;
    /**
     * Header control to render: a text box by default, a value picker, or a
     * distribution to brush a range out of.
     */
    filterVariant?: "text" | "faceted" | "range";
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
  className = "oss-table-filter",
  inputMode,
  onChange,
  placeholder,
  value: externalValue,
}: {
  ariaLabel: string;
  className?: string;
  inputMode?: "decimal";
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
      className={className}
      inputMode={inputMode}
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

/** Width of every filter popover, shared with `.oss-table-facet-popover`. */
const POPOVER_WIDTH = 260;

/**
 * A header button that opens a dismissible panel beneath itself.
 *
 * The panel is positioned as fixed rather than nested in the scroll container,
 * which would otherwise clip it and widen the table's scrollable area.
 * `children` is a function so that a panel — whose contents can be costly to
 * derive from the rows — is built only while it is open.
 */
export function FilterPopover({
  ariaLabel,
  children,
  label,
}: {
  ariaLabel: string;
  children: () => ReactNode;
  label: string;
}) {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ left: number; top: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const reposition = () => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) {
      setAnchor({
        left: Math.min(
          Math.max(8, rect.left),
          window.innerWidth - POPOVER_WIDTH - 8,
        ),
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
        // A narrow column truncates its summary, so keep the whole of it reachable.
        title={label}
        type="button"
      >
        {label}
      </button>
      {open && anchor && (
        <div
          className="oss-table-facet-popover"
          ref={popoverRef}
          role="dialog"
          style={{ left: anchor.left, top: anchor.top }}
        >
          {children()}
        </div>
      )}
    </>
  );
}

/**
 * Multi-select filter listing a column's distinct values with their counts.
 *
 * TanStack computes the counts from rows passing every *other* column's
 * filters, so the options always describe what is still reachable.
 */
export function FacetedFilter<TData>({
  ariaLabel,
  column,
}: {
  ariaLabel: string;
  column: Column<TData, unknown>;
}) {
  const selected = (column.getFilterValue() as unknown[] | undefined) ?? [];
  return (
    <FilterPopover
      ariaLabel={ariaLabel}
      label={
        selected.length === 0 ? "Any" : `${formatNumber(selected.length)} selected`
      }
    >
      {() => <FacetedFilterPanel ariaLabel={ariaLabel} column={column} />}
    </FilterPopover>
  );
}

function FacetedFilterPanel<TData>({
  ariaLabel,
  column,
}: {
  ariaLabel: string;
  column: Column<TData, unknown>;
}) {
  const [query, setQuery] = useState("");

  const selected = (column.getFilterValue() as unknown[] | undefined) ?? [];
  const selectedSet = new Set(selected);
  const facets = column.getFacetedUniqueValues();

  // Counts rank a category's long tail usefully, but an identity column such
  // as a name is easier to scan alphabetically.
  const byLabel = column.columnDef.meta?.facetSort === "label";
  const options = useMemo(
    () =>
      [...facets.entries()]
        .map(([value, count]) => ({ count, label: facetOptionLabel(value), value }))
        .sort((a, b) =>
          byLabel
            ? a.label.localeCompare(b.label)
            : b.count - a.count || a.label.localeCompare(b.label),
        ),
    [byLabel, facets],
  );
  const needle = query.trim().toLocaleLowerCase();
  const matches = needle
    ? options.filter((option) => option.label.toLocaleLowerCase().includes(needle))
    : options;
  const shown = matches.slice(0, MAX_FACET_OPTIONS);
  const showCounts = options.some((option) => option.count > 1);

  const setSelection = (values: unknown[]) =>
    column.setFilterValue(values.length === 0 ? undefined : values);

  return (
    <>
      <input
        aria-label={`Search ${ariaLabel}`}
        autoFocus
        className="oss-table-facet-search"
        onChange={(event: ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)}
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
              {showCounts && (
                <span className="oss-table-facet-count">{formatNumber(count)}</span>
              )}
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
    </>
  );
}

/** Drawn size of a distribution, in the histogram's own coordinate space. */
const HISTOGRAM_WIDTH = 236;
const HISTOGRAM_HEIGHT = 54;

/**
 * Read a column's values in the units its filter box speaks.
 *
 * The rows are those passing every *other* column's filters, so the
 * distribution describes what is still reachable and holds still while its own
 * bounds are dragged.
 */
function filterValues<TData>(
  column: Column<TData, unknown>,
  rows: readonly Row<TData>[],
): (number | null)[] {
  const scale = column.columnDef.meta?.filterScale ?? 1;
  return rows.map((row) => {
    const value = row.getValue(column.id);
    return typeof value === "number" ? value * scale : null;
  });
}

/** Name the span a bar covers, including the outliers an end bar absorbs. */
function describeBin(
  bin: HistogramBin,
  index: number,
  histogram: Histogram,
  format: (value: number) => string,
): string {
  if (index === 0 && histogram.start > histogram.minimum) {
    return `${format(bin.end)} and below`;
  }
  if (index === histogram.bins.length - 1 && histogram.end < histogram.maximum) {
    return `${format(bin.start)} and above`;
  }
  return `${format(bin.start)} to ${format(bin.end)}`;
}

/**
 * A column's distribution, with the bars inside the current bounds picked out.
 *
 * Bar heights use a square-root scale: ski-area metrics are long-tailed enough
 * that a linear one leaves every bar but the first invisible.
 */
function RangeHistogram({
  bounds,
  format,
  histogram,
  onSelect,
}: {
  bounds: NumericRange;
  format: (value: number) => string;
  histogram: Histogram;
  onSelect: (bounds: NumericRange, settled: boolean) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const anchorRef = useRef<number | null>(null);
  const binWidth = HISTOGRAM_WIDTH / histogram.bins.length;

  const binAt = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) {
      return 0;
    }
    const ratio = (clientX - rect.left) / rect.width;
    return Math.min(
      histogram.bins.length - 1,
      Math.max(0, Math.floor(ratio * histogram.bins.length)),
    );
  };

  const select = (clientX: number, settled: boolean) => {
    const anchor = anchorRef.current;
    if (anchor === null) {
      return;
    }
    const other = binAt(clientX);
    onSelect(
      boundsFromEdges(
        histogram.bins[Math.min(anchor, other)].start,
        histogram.bins[Math.max(anchor, other)].end,
        histogram,
      ),
      settled,
    );
  };

  return (
    <svg
      aria-label={`Distribution from ${format(histogram.start)} to ${format(histogram.end)}. Drag across the bars to select a range.`}
      className="oss-table-range-plot"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        anchorRef.current = binAt(event.clientX);
        select(event.clientX, false);
      }}
      onPointerMove={(event) => {
        if (anchorRef.current !== null) {
          select(event.clientX, false);
        }
      }}
      onPointerUp={(event) => {
        select(event.clientX, true);
        anchorRef.current = null;
      }}
      preserveAspectRatio="none"
      ref={svgRef}
      role="img"
      viewBox={`0 0 ${HISTOGRAM_WIDTH} ${HISTOGRAM_HEIGHT}`}
    >
      {histogram.bins.map((bin, index) => {
        const height =
          bin.count === 0
            ? 0
            : Math.max(
                1,
                Math.sqrt(bin.count / histogram.maxCount) * HISTOGRAM_HEIGHT,
              );
        const selected = rangeContains(bounds, (bin.start + bin.end) / 2);
        return (
          <rect
            className={
              selected ? "oss-table-range-bar" : "oss-table-range-bar-excluded"
            }
            height={Math.max(height, 0.75)}
            key={bin.start}
            width={Math.max(binWidth - 1, 0.75)}
            x={((bin.start - histogram.start) / (histogram.end - histogram.start)) * HISTOGRAM_WIDTH}
            y={HISTOGRAM_HEIGHT - height}
          >
            <title>
              {describeBin(bin, index, histogram, format)}:{" "}
              {formatNumber(bin.count)}
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

function RangeFilterPanel<TData>({
  ariaLabel,
  column,
}: {
  ariaLabel: string;
  column: Column<TData, unknown>;
}) {
  const rows = column.getFacetedRowModel().flatRows;
  const values = useMemo(() => filterValues(column, rows), [column, rows]);
  const histogram = useMemo(() => buildHistogram(values), [values]);
  // Bounds are re-derived only when the filter text changes, so the debounced
  // boxes below see a callback identity that settles instead of churning.
  const filterValue = column.getFilterValue();
  const applied = useMemo(() => boundsFromFilter(filterValue), [filterValue]);
  const [dragged, setDragged] = useState<NumericRange | null>(null);
  const bounds = dragged ?? applied;

  const boundsRef = useRef(bounds);
  boundsRef.current = bounds;
  const commit = useCallback(
    (next: NumericRange) => {
      setDragged(null);
      column.setFilterValue(
        histogram === null ? undefined : formatRangeFilter(next, histogram),
      );
    },
    [column, histogram],
  );
  const commitBound = useCallback(
    (edge: "lower" | "upper", text: string) => {
      const trimmed = text.trim();
      const parsed =
        trimmed === ""
          ? edge === "lower"
            ? Number.NEGATIVE_INFINITY
            : Number.POSITIVE_INFINITY
          : Number(trimmed);
      if (Number.isNaN(parsed)) {
        return;
      }
      commit({
        ...boundsRef.current,
        [edge]: parsed,
        // A typed bound reads as inclusive, whatever bracket a brush left.
        [edge === "lower" ? "lowerInclusive" : "upperInclusive"]: true,
      });
    },
    [commit],
  );
  const commitLower = useCallback(
    (text: string) => commitBound("lower", text),
    [commitBound],
  );
  const commitUpper = useCallback(
    (text: string) => commitBound("upper", text),
    [commitBound],
  );

  if (histogram === null) {
    return <p className="oss-table-facet-empty">No values to filter.</p>;
  }

  const format = (value: number) =>
    column.columnDef.meta?.filterFormat?.(value) ??
    formatBound(value, histogram.precision);
  const boundText = (value: number) =>
    Number.isFinite(value) ? String(roundTo(value, histogram.precision)) : "";
  const keptCount = values.filter(
    (value) => value !== null && rangeContains(bounds, value),
  ).length;
  // Name only the end the axis actually stops short of.
  const outliers = [
    histogram.start > histogram.minimum
      ? `down to ${format(histogram.minimum)}`
      : null,
    histogram.end < histogram.maximum ? `up to ${format(histogram.maximum)}` : null,
  ]
    .filter((clause) => clause !== null)
    .join(" and ");

  return (
    <>
      <p className="oss-table-range-summary">
        {formatNumber(keptCount)} of {formatNumber(histogram.valueCount)} in range
        {histogram.missingCount > 0 &&
          ` · ${formatNumber(histogram.missingCount)} missing`}
      </p>
      <RangeHistogram
        bounds={bounds}
        format={format}
        histogram={histogram}
        onSelect={(next, settled) => (settled ? commit(next) : setDragged(next))}
      />
      <div className="oss-table-range-axis">
        <span>{format(histogram.start)}</span>
        <span>{format(histogram.end)}</span>
      </div>
      {outliers !== "" && (
        <p className="oss-table-facet-more">End bars hold outliers {outliers}.</p>
      )}
      <div className="oss-table-range-inputs">
        <DebouncedInput
          ariaLabel={`Minimum ${ariaLabel}`}
          className="oss-table-range-input"
          inputMode="decimal"
          onChange={commitLower}
          placeholder={boundText(histogram.start)}
          value={boundText(bounds.lower)}
        />
        <span aria-hidden="true">to</span>
        <DebouncedInput
          ariaLabel={`Maximum ${ariaLabel}`}
          className="oss-table-range-input"
          inputMode="decimal"
          onChange={commitUpper}
          placeholder={boundText(histogram.end)}
          value={boundText(bounds.upper)}
        />
      </div>
      <div className="oss-table-facet-actions">
        <span className="oss-table-facet-more">Drag the bars to select.</span>
        <button
          disabled={filterValue === undefined}
          onClick={() => commit(UNBOUNDED)}
          type="button"
        >
          Clear
        </button>
      </div>
    </>
  );
}

/** Filter a numeric column by brushing or typing bounds over its distribution. */
export function RangeFilter<TData>({
  ariaLabel,
  column,
}: {
  ariaLabel: string;
  column: Column<TData, unknown>;
}) {
  const bounds = boundsFromFilter(column.getFilterValue());
  const format = column.columnDef.meta?.filterFormat ?? formatBound;
  return (
    <FilterPopover ariaLabel={ariaLabel} label={describeBounds(bounds, format)}>
      {() => <RangeFilterPanel ariaLabel={ariaLabel} column={column} />}
    </FilterPopover>
  );
}

/** The filter control a column's `filterVariant` asks for. */
export function ColumnFilter<TData>({ column }: { column: Column<TData, unknown> }) {
  const ariaLabel = `Filter ${column.id}`;
  switch (column.columnDef.meta?.filterVariant) {
    case "faceted":
      return <FacetedFilter ariaLabel={ariaLabel} column={column} />;
    case "range":
      return <RangeFilter ariaLabel={ariaLabel} column={column} />;
    default:
      return (
        <DebouncedInput
          ariaLabel={ariaLabel}
          onChange={column.setFilterValue}
          placeholder={column.columnDef.meta?.filterPlaceholder}
          value={column.getFilterValue()}
        />
      );
  }
}
