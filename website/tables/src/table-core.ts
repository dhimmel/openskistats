import type { SkiAreaSummary } from "./types";

export const DISTINCT_FIELDS = [
  "ski_area_name",
  "country",
  "region",
  "locality",
] as const satisfies readonly (keyof SkiAreaSummary)[];

export const SUM_FIELDS = [
  "run_count",
  "lift_count",
  "combined_vertical",
  "vertical_drop",
] as const satisfies readonly (keyof SkiAreaSummary)[];

export const WEIGHTED_PERCENT_FIELDS = [
  "bearing_alignment",
  "poleward_affinity",
  "eastward_affinity",
  "run_proportion_4_north",
  "run_proportion_4_east",
  "run_proportion_4_south",
  "run_proportion_4_west",
  "run_proportion_2_north",
] as const satisfies readonly (keyof SkiAreaSummary)[];

type DistinctField = (typeof DISTINCT_FIELDS)[number];
type SumField = (typeof SUM_FIELDS)[number];
type WeightedPercentField = (typeof WEIGHTED_PERCENT_FIELDS)[number];

export interface FilteredAggregates {
  rowCount: number;
  distinctCounts: Record<DistinctField, number>;
  sums: Record<SumField, number | null>;
  minimumElevation: number | null;
  maximumElevation: number | null;
  weightedMeans: Record<WeightedPercentField, number | null>;
}

function nullableSum(
  rows: readonly SkiAreaSummary[],
  field: SumField,
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

export function calculateFilteredAggregates(
  rows: readonly SkiAreaSummary[],
): FilteredAggregates {
  const distinctCounts = Object.fromEntries(
    DISTINCT_FIELDS.map((field) => [
      field,
      new Set(
        rows
          .map((row) => row[field])
          .filter((value) => value !== null && value !== undefined),
      ).size,
    ]),
  ) as Record<DistinctField, number>;

  const sums = Object.fromEntries(
    SUM_FIELDS.map((field) => [field, nullableSum(rows, field)]),
  ) as Record<SumField, number | null>;

  const elevations = rows.flatMap((row) =>
    typeof row.min_elevation === "number" ? [row.min_elevation] : [],
  );
  const peakElevations = rows.flatMap((row) =>
    typeof row.max_elevation === "number" ? [row.max_elevation] : [],
  );

  const weightedMeans = Object.fromEntries(
    WEIGHTED_PERCENT_FIELDS.map((field) => {
      let numerator = 0;
      let denominator = 0;
      for (const row of rows) {
        const value = row[field];
        const weight = row.combined_vertical;
        if (typeof value === "number" && typeof weight === "number" && weight > 0) {
          numerator += value * weight;
          denominator += weight;
        }
      }
      return [field, denominator === 0 ? null : numerator / denominator];
    }),
  ) as Record<WeightedPercentField, number | null>;

  return {
    rowCount: rows.length,
    distinctCounts,
    sums,
    minimumElevation: elevations.length === 0 ? null : Math.min(...elevations),
    maximumElevation:
      peakElevations.length === 0 ? null : Math.max(...peakElevations),
    weightedMeans,
  };
}
