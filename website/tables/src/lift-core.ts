import type { LiftSummary } from "./types";

export const LIFT_DISTINCT_FIELDS = [
  "lift_name",
  "lift_type",
  "country",
  "region",
] as const satisfies readonly (keyof LiftSummary)[];

export const LIFT_SUM_FIELDS = [
  "inclined_length",
  "vertical_rise",
  "lift_capacity",
] as const satisfies readonly (keyof LiftSummary)[];

type LiftDistinctField = (typeof LIFT_DISTINCT_FIELDS)[number];
type LiftSumField = (typeof LIFT_SUM_FIELDS)[number];

export interface LiftAggregates {
  rowCount: number;
  distinctCounts: Record<LiftDistinctField, number>;
  distinctSkiAreas: number;
  sums: Record<LiftSumField, number | null>;
  minimumElevation: number | null;
  maximumElevation: number | null;
  medianDuration: number | null;
}

function nullableSum(
  rows: readonly LiftSummary[],
  field: LiftSumField,
): number | null {
  let count = 0;
  let sum = 0;
  for (const row of rows) {
    const value = row[field];
    if (typeof value === "number") {
      count += 1;
      sum += value;
    }
  }
  return count === 0 ? null : sum;
}

function median(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

/** Summarize the filtered rows in one pass for the table footer. */
export function calculateLiftAggregates(
  rows: readonly LiftSummary[],
): LiftAggregates {
  const distinctCounts = Object.fromEntries(
    LIFT_DISTINCT_FIELDS.map((field) => [
      field,
      new Set(
        rows
          .map((row) => row[field])
          .filter((value) => value !== null && value !== undefined),
      ).size,
    ]),
  ) as Record<LiftDistinctField, number>;

  const sums = Object.fromEntries(
    LIFT_SUM_FIELDS.map((field) => [field, nullableSum(rows, field)]),
  ) as Record<LiftSumField, number | null>;

  const baseElevations = rows.flatMap((row) =>
    typeof row.min_elevation === "number" ? [row.min_elevation] : [],
  );
  const peakElevations = rows.flatMap((row) =>
    typeof row.max_elevation === "number" ? [row.max_elevation] : [],
  );

  return {
    rowCount: rows.length,
    distinctCounts,
    distinctSkiAreas: new Set(rows.flatMap((row) => row.ski_area_ids)).size,
    sums,
    minimumElevation:
      baseElevations.length === 0 ? null : Math.min(...baseElevations),
    maximumElevation:
      peakElevations.length === 0 ? null : Math.max(...peakElevations),
    medianDuration: median(
      rows.flatMap((row) =>
        typeof row.lift_duration === "number" ? [row.lift_duration] : [],
      ),
    ),
  };
}
