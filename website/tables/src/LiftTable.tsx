import {
  flexRender,
  type CellContext,
  type ColumnDef,
  type FilterFn,
  type PaginationState,
  useTable,
} from "@tanstack/react-table";
import { useEffect, useMemo, useState } from "react";

import { matchesNumericFilter, matchesSetFilter } from "./filters";
import {
  formatLatitude,
  formatLongitude,
  formatMeters,
  formatNumber,
  MISSING_VALUE,
} from "./formatters";
import { calculateLiftAggregates, type LiftAggregates } from "./lift-core";
import { TABLE_FEATURES, type TableFeatures } from "./table-features";
import {
  ColumnFilter,
  ColumnVisibilityPicker,
  columnMaximum,
  CountryCell,
  countryColumnMeta,
  footerStat,
  HiddenColumnNotices,
  LatitudeCell,
  LongitudeCell,
  metricCell,
  renderColumnHeader,
  textCell,
} from "./table-ui";
import {
  isIntegerField,
  type LiftDocument,
  type LiftSummary,
  type TableRecordSchema,
} from "./types";
import {
  BOOLEAN_URL_VALUE,
  type TableUrlState,
  urlColumnsFrom,
  useUrlTableState,
  visibilityColumnsFrom,
} from "./url-state";

/** Lifts recorded as still operating, shown before the visitor clears filters. */
export const INITIAL_LIFT_FILTERS = [
  { id: "lift_status", value: ["operating"] },
] as const;

/** The view an untouched table shows, which a shared link leaves unwritten. */
const DEFAULT_STATE: TableUrlState = {
  columnFilters: [...INITIAL_LIFT_FILTERS],
  sorting: [{ desc: true, id: "vertical_rise" }],
  columnVisibility: { longitude: false },
};

const numericFilter: FilterFn<TableFeatures, LiftSummary> = (row, columnId, value) =>
  matchesNumericFilter(row.getValue<number | null>(columnId), value);

/** Keep rows whose value was selected in a column's value picker. */
const setFilter: FilterFn<TableFeatures, LiftSummary> = (row, columnId, value) =>
  matchesSetFilter(row.getValue(columnId), value);

/** Keep lifts serving any of the selected ski areas. */
const skiAreaFilter: FilterFn<TableFeatures, LiftSummary> = (
  row,
  _columnId,
  value,
) => {
  if (!Array.isArray(value) || value.length === 0) {
    return true;
  }
  const selected = new Set(value as string[]);
  return row.original.ski_area_names.some(
    (name) => name !== null && selected.has(name),
  );
};

/** Format a ride duration in seconds as minutes and seconds. */
export function formatDuration(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0) {
    return MISSING_VALUE;
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function SkiAreaCell({ row }: CellContext<TableFeatures, LiftSummary, unknown>) {
  const { ski_area_ids, ski_area_names } = row.original;
  const named = ski_area_names.flatMap((name, index) =>
    name === null ? [] : [{ id: ski_area_ids[index], name }],
  );
  if (named.length === 0) {
    return MISSING_VALUE;
  }
  return (
    <span className="oss-table-ski-areas">
      {named.map(({ id, name }) => (
        <a
          href={`https://openskimap.org/?obj=${id}`}
          key={id}
          rel="noreferrer"
          target="_blank"
        >
          {name}
        </a>
      ))}
    </span>
  );
}

/** Render a nullable boolean as a check, a dash, or a missing marker. */
function BooleanCell({
  getValue,
}: CellContext<TableFeatures, LiftSummary, unknown>) {
  const value = getValue<boolean | null>();
  if (value === null) {
    return <span className="oss-table-boolean-unknown">{MISSING_VALUE}</span>;
  }
  return <span aria-label={value ? "yes" : "no"}>{value ? "✓" : "·"}</span>;
}

function aggregatesFrom(context: {
  table: { options: { meta?: { aggregates: unknown } } };
}): LiftAggregates | undefined {
  return context.table.options.meta?.aggregates as LiftAggregates | undefined;
}

function createColumns(
  data: readonly LiftSummary[],
  schema: TableRecordSchema,
): ColumnDef<TableFeatures, LiftSummary, unknown>[] {
  const numericColumn = (
    field: keyof LiftSummary,
    label: string,
    options: Partial<ColumnDef<TableFeatures, LiftSummary, unknown>> = {},
  ): ColumnDef<TableFeatures, LiftSummary, unknown> => ({
    accessorKey: field,
    filterFn: numericFilter,
    header: label,
    sortUndefined: "last",
    ...options,
    id: field,
    meta: {
      filterVariant: "range",
      integer: isIntegerField(schema, field),
      ...options.meta,
    },
  });
  const categoricalColumn = (
    field: keyof LiftSummary,
    label: string,
    options: Partial<ColumnDef<TableFeatures, LiftSummary, unknown>> = {},
  ): ColumnDef<TableFeatures, LiftSummary, unknown> => ({
    accessorKey: field,
    cell: ({ getValue }) => textCell(getValue<string | null>()),
    filterFn: setFilter,
    header: label,
    sortDescFirst: false,
    sortUndefined: "last",
    ...options,
    id: field,
    meta: { filterVariant: "faceted", ...options.meta },
  });

  return [
    {
      header: "",
      id: "lift-group",
      meta: { className: "oss-table-sticky" },
      columns: [
        {
          accessorKey: "lift_name",
          enableHiding: false,
          cell: ({ getValue, row }) => (
            <a
              href={`https://openskimap.org/?obj=${row.original.lift_id}`}
              rel="noreferrer"
              target="_blank"
            >
              {getValue<string>()}
            </a>
          ),
          filterFn: setFilter,
          footer: (context) =>
            footerStat(
              "Lifts",
              formatNumber(aggregatesFrom(context)?.rowCount ?? null),
            ),
          header: "Lift",
          id: "lift_name",
          meta: { facetSort: "label", filterVariant: "faceted" },
          minSize: 140,
          size: 165,
          sortDescFirst: false,
        },
      ],
    },
    {
      header: "Location",
      meta: { className: "oss-table-border-left" },
      columns: [
        {
          accessorKey: "ski_area_names",
          cell: SkiAreaCell,
          filterFn: skiAreaFilter,
          footer: (context) =>
            footerStat(
              "Distinct",
              formatNumber(aggregatesFrom(context)?.distinctSkiAreas ?? null),
            ),
          header: "Ski Area",
          id: "ski_area_names",
          getUniqueValues: (lift) =>
            [...new Set(lift.ski_area_names)].filter(
              (name): name is string => name !== null,
            ),
          meta: {
            className: "oss-table-border-left",
            filterVariant: "faceted",
          },
          minSize: 110,
          size: 135,
          sortDescFirst: false,
          // Sort by the first associated ski area, since a lift may serve several.
          sortFn: (rowA, rowB) =>
            (rowA.original.ski_area_names[0] ?? "").localeCompare(
              rowB.original.ski_area_names[0] ?? "",
            ),
          sortUndefined: "last",
        },
        categoricalColumn("country", "Country", {
          cell: CountryCell,
          meta: countryColumnMeta(data),
          footer: (context) =>
            footerStat(
              "Distinct",
              formatNumber(aggregatesFrom(context)?.distinctCounts.country ?? null),
            ),
          minSize: 70,
          size: 85,
        }),
        categoricalColumn("region", "Region", {
          footer: (context) =>
            footerStat(
              "Distinct",
              formatNumber(aggregatesFrom(context)?.distinctCounts.region ?? null),
            ),
          minSize: 65,
          size: 82,
        }),
        categoricalColumn("locality", "Locality", {
          minSize: 65,
          size: 82,
        }),
        {
          accessorKey: "latitude",
          cell: LatitudeCell,
          filterFn: numericFilter,
          header: "ℍ φ",
          id: "latitude",
          meta: { filterFormat: formatLatitude, filterVariant: "range" },
          minSize: 55,
          size: 62,
          sortUndefined: "last",
        },
        numericColumn("longitude", "ℍ λ", {
          cell: LongitudeCell,
          meta: { filterFormat: formatLongitude },
          minSize: 55,
          size: 62,
        }),
      ],
    },
    {
      header: "Lift",
      meta: { className: "oss-table-border-left" },
      columns: [
        categoricalColumn("lift_type", "Type", {
          footer: (context) =>
            footerStat(
              "Distinct",
              formatNumber(aggregatesFrom(context)?.distinctCounts.lift_type ?? null),
            ),
          meta: { className: "oss-table-border-left", filterVariant: "faceted" },
          minSize: 70,
          size: 85,
        }),
        categoricalColumn("lift_status", "Status", {
          minSize: 65,
          size: 78,
        }),
        {
          accessorKey: "lift_detachable",
          cell: BooleanCell,
          filterFn: setFilter,
          header: "Detach.",
          id: "lift_detachable",
          meta: { filterVariant: "faceted", urlValue: BOOLEAN_URL_VALUE },
          minSize: 50,
          size: 58,
          sortUndefined: "last",
        },
        numericColumn("lift_occupancy", "Seats", {
          minSize: 48,
          size: 55,
        }),
        numericColumn("lift_capacity", "Capacity", {
          cell: metricCell(columnMaximum(data, "lift_capacity")),
          footer: (context) =>
            footerStat(
              "Sum",
              formatNumber(aggregatesFrom(context)?.sums.lift_capacity ?? null),
            ),
          minSize: 60,
          size: 72,
        }),
        numericColumn("lift_duration", "Ride", {
          cell: ({ getValue }) => formatDuration(getValue<number | null>()),
          footer: (context) =>
            footerStat(
              "Median",
              formatDuration(aggregatesFrom(context)?.medianDuration ?? null),
            ),
          // Bounds are seconds, but a reader recognises them as minutes.
          meta: { filterFormat: formatDuration },
          minSize: 50,
          size: 60,
        }),
      ],
    },
    {
      header: "Dimensions",
      meta: { className: "oss-table-border-left" },
      columns: [
        numericColumn("inclined_length", "Length", {
          cell: metricCell(columnMaximum(data, "inclined_length"), formatMeters),
          footer: (context) =>
            footerStat(
              "Sum",
              formatMeters(aggregatesFrom(context)?.sums.inclined_length ?? null),
            ),
          meta: { className: "oss-table-border-left" },
          minSize: 62,
          size: 76,
        }),
        numericColumn("vertical_rise", "Rise", {
          cell: metricCell(columnMaximum(data, "vertical_rise"), formatMeters),
          footer: (context) =>
            footerStat(
              "Sum",
              formatMeters(aggregatesFrom(context)?.sums.vertical_rise ?? null),
            ),
          minSize: 58,
          size: 70,
        }),
        numericColumn("min_elevation", "Base Elev", {
          cell: metricCell(columnMaximum(data, "min_elevation"), formatMeters),
          footer: (context) =>
            footerStat(
              "Min",
              formatMeters(aggregatesFrom(context)?.minimumElevation ?? null),
            ),
          minSize: 60,
          size: 72,
        }),
        numericColumn("max_elevation", "Peak Elev", {
          cell: metricCell(columnMaximum(data, "max_elevation"), formatMeters),
          footer: (context) =>
            footerStat(
              "Max",
              formatMeters(aggregatesFrom(context)?.maximumElevation ?? null),
            ),
          minSize: 60,
          size: 72,
        }),
      ],
    },
  ];
}

export function LiftTable({ document }: { document: LiftDocument }) {
  const columns = useMemo(
    () => createColumns(document.lifts, document.record_schema),
    [document],
  );
  const urlSpec = useMemo(
    () => ({ columns: urlColumnsFrom(columns), visibilityColumns: visibilityColumnsFrom(columns), defaults: DEFAULT_STATE }),
    [columns],
  );
  const { columnFilters, columnVisibility, setColumnFilters, setColumnVisibility, setSorting, sorting } =
    useUrlTableState(urlSpec);
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 10,
  });
  const [aggregates, setAggregates] = useState(() =>
    calculateLiftAggregates(document.lifts),
  );

  const table = useTable({
    features: TABLE_FEATURES,
    columns,
    data: document.lifts,
    defaultColumn: { maxSize: 190, minSize: 36, size: 55 },
    initialState: { columnVisibility: DEFAULT_STATE.columnVisibility },
    meta: { aggregates },
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    onPaginationChange: setPagination,
    onSortingChange: setSorting,
    state: { columnFilters, columnVisibility, pagination, sorting },
  });

  const filteredRows = table.getFilteredRowModel().rows;
  useEffect(() => {
    setAggregates(calculateLiftAggregates(filteredRows.map((row) => row.original)));
  }, [filteredRows]);

  const pageCount = table.getPageCount();
  const hasFilters = columnFilters.length > 0;

  return (
    <div className="oss-table">
      <div className="oss-table-status" role="status">
        <span>
          Showing {formatNumber(filteredRows.length)} of {formatNumber(document.record_count)} named lifts.
        </span>
        <div className="oss-table-actions">
          <ColumnVisibilityPicker table={table} />
          <button
            className="oss-table-clear"
            disabled={!hasFilters}
            onClick={() => setColumnFilters([])}
            type="button"
          >
            Clear all filters
          </button>
        </div>
      </div>
      <HiddenColumnNotices table={table} />
      <div className="oss-table-scroll" tabIndex={0}>
        <table style={{ minWidth: table.getTotalSize(), width: "100%" }}>
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
                    className={`${headerCell.column.columnDef.meta?.className ?? ""} ${headerCell.column.id === "lift_name" ? "oss-table-sticky" : ""}`}
                    colSpan={headerCell.colSpan}
                    key={headerCell.id}
                    scope={headerCell.colSpan > 1 ? "colgroup" : "col"}
                    style={{ width: headerCell.getSize() }}
                  >
                    {headerCell.isPlaceholder
                      ? null
                      : renderColumnHeader(
                          headerCell.getContext(),
                          document.record_schema.properties[headerCell.column.id]?.description,
                        )}
                    {!headerCell.isPlaceholder &&
                      headerCell.colSpan === 1 &&
                      headerCell.column.getCanFilter() && (
                        <ColumnFilter column={headerCell.column} />
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
                  No lifts match the current filters. Clear or adjust a filter to continue.
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td
                      className={`${cell.column.columnDef.meta?.className ?? ""} ${cell.column.id === "lift_name" ? "oss-table-sticky" : ""}`}
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
                    className={`${footerCell.column.columnDef.meta?.className ?? ""} ${footerCell.column.id === "lift_name" ? "oss-table-sticky" : ""}`}
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
