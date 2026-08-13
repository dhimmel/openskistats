import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type CellContext,
  type ColumnDef,
  type ColumnFiltersState,
  type FilterFn,
  type HeaderContext,
  type PaginationState,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  type CSSProperties,
  type ChangeEvent,
  type FocusEvent,
  type PointerEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  countryCodeToFlag,
  INITIAL_COLUMN_FILTERS,
  matchesCountryFilter,
  matchesLatitudeFilter,
  matchesNumericFilter,
  matchesPercentFilter,
} from "./filters";
import {
  formatMeters,
  formatNumber,
  formatPercent,
  MISSING_VALUE,
} from "./formatters";
import {
  calculateFilteredAggregates,
  type FilteredAggregates,
} from "./table-core";
import type {
  SkiAreaDocument,
  SkiAreaRecordSchema,
  SkiAreaSummary,
} from "./types";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends unknown, TValue> {
    cellStyle?: (value: unknown) => CSSProperties;
    className?: string;
    filterPlaceholder?: string;
  }

  interface TableMeta<TData extends unknown> {
    aggregates: FilteredAggregates;
  }
}

const numericFilter: FilterFn<SkiAreaSummary> = (row, columnId, value) =>
  matchesNumericFilter(row.getValue<number | null>(columnId), value);

const percentFilter: FilterFn<SkiAreaSummary> = (row, columnId, value) =>
  matchesPercentFilter(row.getValue<number | null>(columnId), value);

const countryFilter: FilterFn<SkiAreaSummary> = (row, _columnId, value) =>
  matchesCountryFilter(row.original, value);

const latitudeFilter: FilterFn<SkiAreaSummary> = (row, columnId, value) =>
  matchesLatitudeFilter(row.getValue<number | null>(columnId), value);

function fieldDescription(
  schema: SkiAreaRecordSchema,
  field: keyof SkiAreaSummary,
): string | undefined {
  return schema.properties[field]?.description;
}

function HeaderLabel({
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

function header(
  label: ReactNode,
  description?: string,
): (context: HeaderContext<SkiAreaSummary, unknown>) => ReactNode {
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

function DebouncedInput({
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

function textCell(value: string | null): ReactNode {
  return value ?? MISSING_VALUE;
}

function CountryCell({ row }: CellContext<SkiAreaSummary, unknown>) {
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

function LatitudeCell({ getValue }: CellContext<SkiAreaSummary, unknown>) {
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

function AzimuthCell({ getValue }: CellContext<SkiAreaSummary, unknown>) {
  const azimuth = getValue<number | null>();
  if (azimuth === null) {
    return MISSING_VALUE;
  }
  return (
    <span className="oss-table-azimuth">
      <svg
        aria-hidden="true"
        className="oss-table-azimuth-arrow"
        style={{ transform: `rotate(${azimuth}deg)` }}
        viewBox="0 0 36 36"
      >
        <circle cx="18" cy="18" fill="currentColor" r="3" />
        <line stroke="currentColor" strokeWidth="3" x1="18" x2="18" y1="18" y2="6" />
        <polygon fill="currentColor" points="12,9 18,1.5 24,9" />
      </svg>
      <span>{formatNumber(azimuth)}°</span>
    </span>
  );
}

function DonutCell({ getValue }: CellContext<SkiAreaSummary, unknown>) {
  const value = getValue<number | null>();
  if (value === null) {
    return MISSING_VALUE;
  }
  const radius = 24;
  const circumference = 2 * Math.PI * radius;
  return (
    <span aria-label={formatPercent(value)} className="oss-table-donut">
      <svg aria-hidden="true" viewBox="0 0 60 60">
        <circle className="oss-table-donut-track" cx="30" cy="30" r={radius} />
        <circle
          className="oss-table-donut-value"
          cx="30"
          cy="30"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value)}
        />
      </svg>
      <span>{formatPercent(value)}</span>
    </span>
  );
}

function roseTooltipPosition(element: HTMLElement) {
  const rect = element.getBoundingClientRect();
  const size = 300;
  const gap = 10;
  const left = Math.max(gap, rect.left - size - gap);
  const top = Math.max(
    gap,
    Math.min(
      Math.max(gap, rect.top + rect.height / 2 - size / 2),
      Math.max(gap, window.innerHeight - size - gap),
    ),
  );
  return { left, top };
}

function RoseCell({ row }: CellContext<SkiAreaSummary, unknown>) {
  const [tooltip, setTooltip] = useState<{ left: number; top: number } | null>(null);
  const id = row.original.ski_area_id;
  const previewUrl = `/ski-areas/roses-preview/${id}.svg`;
  const fullUrl = `/ski-areas/roses-full/${id}.svg`;
  const showTooltip = (
    event: PointerEvent<HTMLAnchorElement> | FocusEvent<HTMLAnchorElement>,
  ) => setTooltip(roseTooltipPosition(event.currentTarget));

  return (
    <span className="oss-table-rose">
      <a
        aria-label={`Open full orientation rose for ${row.original.ski_area_name}`}
        href={fullUrl}
        onBlur={() => setTooltip(null)}
        onFocus={showTooltip}
        onPointerEnter={showTooltip}
        onPointerLeave={() => setTooltip(null)}
        rel="noreferrer"
        target="_blank"
      >
        <img
          alt={`Orientation rose preview for ${row.original.ski_area_name}`}
          loading="lazy"
          src={previewUrl}
        />
      </a>
      {tooltip && (
        <span
          className="oss-table-rose-tooltip"
          style={{ left: tooltip.left, top: tooltip.top }}
        >
          <img alt="" src={fullUrl} />
        </span>
      )}
    </span>
  );
}

function interpolateColor(start: [number, number, number], end: [number, number, number], value: number) {
  const amount = Math.min(1, Math.max(0, value));
  return `rgb(${start.map((channel, index) => Math.round(channel + (end[index] - channel) * amount)).join(", ")})`;
}

function sequentialColor(value: number) {
  return interpolateColor([255, 255, 255], [161, 0, 191], value);
}

function divergingColor(value: number) {
  return value <= 0
    ? interpolateColor([232, 146, 0], [255, 255, 255], value + 1)
    : interpolateColor([255, 255, 255], [0, 125, 191], value);
}

function metricCell(
  maximum: number,
  formatter: (value: number | null) => string = formatNumber,
) {
  return ({ getValue }: CellContext<SkiAreaSummary, unknown>) => {
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

function percentCell() {
  return ({ getValue }: CellContext<SkiAreaSummary, unknown>) => {
    const value = getValue<number | null>();
    return <span>{formatPercent(value)}</span>;
  };
}

function maximum(data: readonly SkiAreaSummary[], field: keyof SkiAreaSummary) {
  return Math.max(
    0,
    ...data.flatMap((row) =>
      typeof row[field] === "number" ? [row[field] as number] : [],
    ),
  );
}

function aggregatesFrom(context: { table: { options: { meta?: { aggregates: FilteredAggregates } } } }) {
  return context.table.options.meta?.aggregates;
}

function createColumns(
  data: readonly SkiAreaSummary[],
  schema: SkiAreaRecordSchema,
): ColumnDef<SkiAreaSummary, unknown>[] {
  const description = (field: keyof SkiAreaSummary) => fieldDescription(schema, field);
  const fieldMaximum = (field: keyof SkiAreaSummary) => maximum(data, field);
  const numericColumn = (
    field: keyof SkiAreaSummary,
    label: ReactNode,
    options: Partial<ColumnDef<SkiAreaSummary, unknown>> = {},
  ): ColumnDef<SkiAreaSummary, unknown> => ({
    accessorKey: field,
    filterFn: numericFilter,
    header: header(label, description(field)),
    minSize: 36,
    sortUndefined: "last",
    ...options,
    id: field,
    meta: { filterPlaceholder: "Number or range", ...options.meta },
  });
  const percentColumn = (
    field: keyof SkiAreaSummary,
    label: ReactNode,
    options: Partial<ColumnDef<SkiAreaSummary, unknown>> = {},
  ): ColumnDef<SkiAreaSummary, unknown> => ({
    accessorKey: field,
    cell: percentCell(),
    filterFn: percentFilter,
    header: header(label, description(field)),
    minSize: 36,
    sortUndefined: "last",
    ...options,
    id: field,
    meta: {
      cellStyle: (value) =>
        typeof value === "number"
          ? { backgroundColor: sequentialColor(value) }
          : {},
      filterPlaceholder: "Percent or range",
      ...options.meta,
    },
  });

  return [
    {
      header: "",
      id: "ski-area-group",
      meta: { className: "oss-table-sticky" },
      columns: [
        {
          accessorKey: "ski_area_id",
          enableColumnFilter: false,
          id: "ski_area_id",
        },
        {
          accessorKey: "ski_area_name",
          cell: ({ getValue, row }) => (
            <a
              href={`https://openskimap.org/?obj=${row.original.ski_area_id}`}
              rel="noreferrer"
              target="_blank"
            >
              {getValue<string>()}
            </a>
          ),
          footer: (context) =>
            `Distinct: ${formatNumber(aggregatesFrom(context)?.distinctCounts.ski_area_name ?? null)}`,
          header: header("Ski Area", description("ski_area_name")),
          id: "ski_area_name",
          minSize: 110,
          size: 130,
          sortDescFirst: false,
        },
        {
          accessorKey: "osm_status",
          id: "osm_status",
        },
      ],
    },
    {
      header: "Location",
      meta: { className: "oss-table-border-left" },
      columns: [
        {
          accessorKey: "latitude",
          cell: LatitudeCell,
          filterFn: latitudeFilter,
          header: header("ℍ φ", description("latitude")),
          id: "latitude",
          meta: { filterPlaceholder: "Latitude or hemisphere" },
          minSize: 50,
          size: 55,
          sortUndefined: "last",
        },
        {
          accessorKey: "country",
          cell: CountryCell,
          filterFn: countryFilter,
          footer: (context) =>
            `Distinct: ${formatNumber(aggregatesFrom(context)?.distinctCounts.country ?? null)}`,
          header: header("Country", description("country")),
          id: "country",
          meta: { className: "oss-table-border-left", filterPlaceholder: "Name, code, or flag" },
          minSize: 60,
          size: 70,
          sortDescFirst: false,
          sortUndefined: "last",
        },
        {
          accessorKey: "country_code",
          id: "country_code",
        },
        {
          accessorKey: "region",
          cell: ({ getValue }) => textCell(getValue<string | null>()),
          footer: (context) =>
            `Distinct: ${formatNumber(aggregatesFrom(context)?.distinctCounts.region ?? null)}`,
          header: header("Region", description("region")),
          id: "region",
          minSize: 55,
          size: 65,
          sortDescFirst: false,
          sortUndefined: "last",
        },
        {
          accessorKey: "locality",
          cell: ({ getValue }) => textCell(getValue<string | null>()),
          footer: (context) =>
            `Distinct: ${formatNumber(aggregatesFrom(context)?.distinctCounts.locality ?? null)}`,
          header: header("Locality", description("locality")),
          id: "locality",
          minSize: 55,
          size: 65,
          sortDescFirst: false,
          sortUndefined: "last",
        },
      ],
    },
    {
      header: "Downhill Runs",
      meta: { className: "oss-table-border-left" },
      columns: [
        numericColumn("run_count", "Runs", {
          cell: metricCell(fieldMaximum("run_count")),
          footer: (context) =>
            `Sum: ${formatNumber(aggregatesFrom(context)?.sums.run_count ?? null)}`,
          meta: { className: "oss-table-border-left" },
          size: 40,
        }),
        numericColumn("lift_count", "Lifts", {
          cell: metricCell(fieldMaximum("lift_count")),
          footer: (context) =>
            `Sum: ${formatNumber(aggregatesFrom(context)?.sums.lift_count ?? null)}`,
          size: 40,
        }),
        numericColumn("combined_vertical", "Vertical", {
          cell: metricCell(fieldMaximum("combined_vertical"), formatMeters),
          footer: (context) =>
            `Sum: ${formatMeters(aggregatesFrom(context)?.sums.combined_vertical ?? null)}`,
          minSize: 55,
          size: 65,
        }),
        numericColumn("min_elevation", "Base Elev", {
          cell: metricCell(fieldMaximum("min_elevation"), formatMeters),
          footer: (context) =>
            `Min: ${formatMeters(aggregatesFrom(context)?.minimumElevation ?? null)}`,
          minSize: 50,
          size: 55,
        }),
        numericColumn("max_elevation", "Peak Elev", {
          cell: metricCell(fieldMaximum("max_elevation"), formatMeters),
          footer: (context) =>
            `Max: ${formatMeters(aggregatesFrom(context)?.maximumElevation ?? null)}`,
          minSize: 50,
          size: 55,
        }),
        numericColumn("vertical_drop", "Drop", {
          cell: metricCell(fieldMaximum("vertical_drop"), formatMeters),
          footer: (context) =>
            `Sum: ${formatMeters(aggregatesFrom(context)?.sums.vertical_drop ?? null)}`,
          minSize: 50,
          size: 55,
        }),
        numericColumn("solar_irradiation_season", "Sunlight", {
          cell: metricCell(fieldMaximum("solar_irradiation_season"), (value) =>
            formatNumber(value, 1),
          ),
          minSize: 45,
          size: 50,
        }),
      ],
    },
    {
      header: "Mean Orientation",
      meta: { className: "oss-table-border-left" },
      columns: [
        numericColumn("bearing_mean", "Azimuth", {
          cell: AzimuthCell,
          meta: { className: "oss-table-border-left" },
          minSize: 45,
          size: 50,
        }),
        percentColumn("bearing_alignment", "Alignment", {
          cell: DonutCell,
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.bearing_alignment ?? null)}`,
          meta: { cellStyle: () => ({}) },
          minSize: 55,
          size: 65,
        }),
        percentColumn("poleward_affinity", "Poleward", {
          cell: percentCell(),
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.poleward_affinity ?? null)}`,
          meta: {
            cellStyle: (value) =>
              typeof value === "number"
                ? { backgroundColor: divergingColor(value) }
                : {},
          },
          minSize: 55,
          size: 60,
        }),
        percentColumn("eastward_affinity", "Eastward", {
          cell: percentCell(),
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.eastward_affinity ?? null)}`,
          meta: {
            cellStyle: (value) =>
              typeof value === "number"
                ? { backgroundColor: divergingColor(value) }
                : {},
          },
          minSize: 55,
          size: 60,
        }),
      ],
    },
    {
      header: "Cardinal Directions",
      meta: { className: "oss-table-border-left" },
      columns: [
        percentColumn("run_proportion_4_north", "N₄", {
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.run_proportion_4_north ?? null)}`,
          meta: { className: "oss-table-border-left" },
          size: 40,
        }),
        percentColumn("run_proportion_4_east", "E₄", {
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.run_proportion_4_east ?? null)}`,
          size: 40,
        }),
        percentColumn("run_proportion_4_south", "S₄", {
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.run_proportion_4_south ?? null)}`,
          size: 40,
        }),
        percentColumn("run_proportion_4_west", "W₄", {
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.run_proportion_4_west ?? null)}`,
          size: 40,
        }),
      ],
    },
    {
      header: "",
      id: "north-south-group",
      meta: { className: "oss-table-border-left" },
      columns: [
        percentColumn("run_proportion_2_north", "N₂", {
          footer: (context) =>
            `Wtd. Mean: ${formatPercent(aggregatesFrom(context)?.weightedMeans.run_proportion_2_north ?? null)}`,
          meta: { className: "oss-table-border-left" },
          size: 40,
        }),
      ],
    },
    {
      header: "",
      id: "rose-group",
      meta: { className: "oss-table-border-left" },
      columns: [
        {
          cell: RoseCell,
          enableColumnFilter: false,
          enableSorting: false,
          header: () => (
            <HeaderLabel
              description="Preview of the ski area's run-orientation rose. Hover or focus for the full rose, or activate the link to open it."
              label="Rose"
            />
          ),
          id: "rose",
          meta: { className: "oss-table-border-left" },
          minSize: 50,
          size: 55,
        },
      ],
    },
  ];
}

export function SkiAreaTable({ document }: { document: SkiAreaDocument }) {
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>(() =>
    INITIAL_COLUMN_FILTERS.map((filter) => ({ ...filter })),
  );
  const [sorting, setSorting] = useState<SortingState>([
    { id: "combined_vertical", desc: true },
  ]);
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 10,
  });
  const [aggregates, setAggregates] = useState(() =>
    calculateFilteredAggregates(document.ski_areas),
  );

  const columns = useMemo(
    () => createColumns(document.ski_areas, document.record_schema),
    [document],
  );
  const table = useReactTable({
    columns,
    data: document.ski_areas,
    defaultColumn: {
      maxSize: 190,
      minSize: 36,
      size: 55,
    },
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    initialState: {
      columnVisibility: {
        country_code: false,
        osm_status: false,
        ski_area_id: false,
      },
    },
    meta: { aggregates },
    onColumnFiltersChange: setColumnFilters,
    onPaginationChange: setPagination,
    onSortingChange: setSorting,
    state: { columnFilters, pagination, sorting },
  });

  const filteredRows = table.getFilteredRowModel().rows;
  useEffect(() => {
    setAggregates(calculateFilteredAggregates(filteredRows.map((row) => row.original)));
  }, [filteredRows]);

  const pageCount = table.getPageCount();
  const hasFilters = columnFilters.length > 0;

  return (
    <div className="oss-table">
      <div className="oss-table-status" role="status">
        <span>
          Showing {formatNumber(filteredRows.length)} of {formatNumber(document.record_count)} named ski areas.
        </span>
        <button
          className="oss-table-clear"
          disabled={!hasFilters}
          onClick={() => setColumnFilters([])}
          type="button"
        >
          Clear all filters
        </button>
      </div>
      <div className="oss-table-scroll" tabIndex={0}>
        <table style={{ width: table.getTotalSize() }}>
          <colgroup>
            {table.getVisibleLeafColumns().map((column) => (
              <col key={column.id} style={{ width: column.getSize() }} />
            ))}
          </colgroup>
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((headerCell) => (
                  <th
                    className={`${headerCell.column.columnDef.meta?.className ?? ""} ${headerCell.column.id === "ski_area_name" ? "oss-table-sticky" : ""}`}
                    colSpan={headerCell.colSpan}
                    key={headerCell.id}
                    scope={headerCell.colSpan > 1 ? "colgroup" : "col"}
                    style={{ width: headerCell.getSize() }}
                  >
                    {headerCell.isPlaceholder
                      ? null
                      : flexRender(
                          headerCell.column.columnDef.header,
                          headerCell.getContext(),
                        )}
                    {!headerCell.isPlaceholder &&
                      headerCell.colSpan === 1 &&
                      headerCell.column.getCanFilter() && (
                        <DebouncedInput
                          ariaLabel={`Filter ${headerCell.column.id}`}
                          onChange={headerCell.column.setFilterValue}
                          placeholder={headerCell.column.columnDef.meta?.filterPlaceholder}
                          value={headerCell.column.getFilterValue()}
                        />
                      )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td className="oss-table-empty" colSpan={table.getVisibleLeafColumns().length}>
                  No ski areas match the current filters. Clear or adjust a filter to continue.
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td
                      className={`${cell.column.columnDef.meta?.className ?? ""} ${cell.column.id === "ski_area_name" ? "oss-table-sticky" : ""}`}
                      key={cell.id}
                      style={cell.column.columnDef.meta?.cellStyle?.(cell.getValue())}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
          <tfoot>
            {table.getFooterGroups().slice(0, 1).map((footerGroup) => (
              <tr key={footerGroup.id}>
                {footerGroup.headers.map((footerCell) => (
                  <td
                    className={`${footerCell.column.columnDef.meta?.className ?? ""} ${footerCell.column.id === "ski_area_name" ? "oss-table-sticky" : ""}`}
                    colSpan={footerCell.colSpan}
                    key={footerCell.id}
                  >
                    {footerCell.isPlaceholder
                      ? null
                      : flexRender(
                          footerCell.column.columnDef.footer,
                          footerCell.getContext(),
                        )}
                  </td>
                ))}
              </tr>
            ))}
          </tfoot>
        </table>
      </div>
      <div className="oss-table-pagination">
        <div className="oss-table-page-buttons">
          <button
            aria-label="First page"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.firstPage()}
            type="button"
          >
            «
          </button>
          <button
            aria-label="Previous page"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
            type="button"
          >
            ‹
          </button>
          <span>
            Page {formatNumber(pageCount === 0 ? 0 : pagination.pageIndex + 1)} of {formatNumber(pageCount)}
          </span>
          <button
            aria-label="Next page"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
            type="button"
          >
            ›
          </button>
          <button
            aria-label="Last page"
            disabled={!table.getCanNextPage()}
            onClick={() => table.lastPage()}
            type="button"
          >
            »
          </button>
        </div>
        <label>
          Rows per page
          <select
            onChange={(event) => table.setPageSize(Number(event.target.value))}
            value={pagination.pageSize}
          >
            {[10, 25, 50, 100].map((pageSize) => (
              <option key={pageSize} value={pageSize}>
                {pageSize}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
