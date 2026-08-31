import {
  columnFacetingFeature,
  columnFilteringFeature,
  columnSizingFeature,
  columnVisibilityFeature,
  createFacetedRowModel,
  createFacetedUniqueValues,
  createFilteredRowModel,
  createPaginatedRowModel,
  createSortedRowModel,
  metaHelper,
  rowPaginationFeature,
  rowSortingFeature,
  tableFeatures,
} from "@tanstack/react-table";
import type { CSSProperties } from "react";

interface TableColumnMeta {
  cellStyle?: (value: unknown) => CSSProperties;
  className?: string;
  /** Extra strings a facet option can be found by, such as a country's code. */
  facetKeys?: (value: unknown) => readonly string[];
  /** Order a value picker's options: by descending count, or by label. */
  facetSort?: "count" | "label";
  /**
   * Multiply values by this before binning and filtering, so a column stored
   * as a fraction filters in the whole percent its cells display.
   */
  filterScale?: number;
  /** Format a bound for the range popover's labels and summary. */
  filterFormat?: (value: number) => string;
  /** Header control to render: a value picker, or a brushable distribution. */
  filterVariant?: "faceted" | "range";
}

interface TableMeta {
  /** Each table supplies its own aggregate shape and narrows it when reading. */
  aggregates: unknown;
}

/** Features shared by both interactive tables, registered explicitly for V9. */
export const TABLE_FEATURES = tableFeatures({
  columnFacetingFeature,
  columnFilteringFeature,
  columnSizingFeature,
  columnVisibilityFeature,
  rowPaginationFeature,
  rowSortingFeature,
  facetedRowModel: createFacetedRowModel(),
  facetedUniqueValues: createFacetedUniqueValues(),
  filteredRowModel: createFilteredRowModel(),
  paginatedRowModel: createPaginatedRowModel(),
  sortedRowModel: createSortedRowModel(),
  columnMeta: metaHelper<TableColumnMeta>(),
  tableMeta: metaHelper<TableMeta>(),
});

export type TableFeatures = typeof TABLE_FEATURES;
