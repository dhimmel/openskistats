import { describe, expect, it } from "vitest";

import { calculateFilteredAggregates } from "../src/table-core";
import type { SkiAreaSummary } from "../src/types";

function skiArea(overrides: Partial<SkiAreaSummary>): SkiAreaSummary {
  return {
    ski_area_id: "id",
    ski_area_name: "Example",
    osm_status: "operating",
    osm_run_convention: "north_america",
    ski_area_uses: ["downhill"],
    country: "United States",
    country_code: "US",
    country_subdiv_code: "US-NH",
    region: "New Hampshire",
    locality: "Exampleville",
    latitude: 44,
    longitude: -72,
    ski_area_websites: null,
    ski_area_sources: null,
    wikidata_id: null,
    run_count: 10,
    lift_count: 2,
    combined_vertical: 100,
    combined_distance: 1_000,
    vertical_drop: 50,
    min_elevation: 200,
    max_elevation: 500,
    solar_irradiation_season: 2.3,
    bearing_mean: 0,
    bearing_alignment: 0.5,
    poleward_affinity: 0.2,
    eastward_affinity: 0.1,
    run_proportion_4_north: 0.4,
    run_proportion_4_east: 0.3,
    run_proportion_4_south: 0.2,
    run_proportion_4_west: 0.1,
    run_proportion_2_north: 0.6,
    ...overrides,
  };
}

describe("calculateFilteredAggregates", () => {
  it("calculates all footer values over the supplied filtered rows", () => {
    const result = calculateFilteredAggregates([
      skiArea({ ski_area_id: "a", combined_vertical: 100, bearing_alignment: 0.25 }),
      skiArea({
        ski_area_id: "b",
        ski_area_name: "Other",
        country: "Canada",
        country_code: "CA",
        combined_vertical: 300,
        bearing_alignment: 0.75,
        run_count: 20,
        lift_count: 4,
        vertical_drop: 75,
        min_elevation: 100,
        max_elevation: 700,
      }),
    ]);

    expect(result.rowCount).toBe(2);
    expect(result.distinctCounts.ski_area_name).toBe(2);
    expect(result.distinctCounts.country).toBe(2);
    expect(result.sums).toMatchObject({
      run_count: 30,
      lift_count: 6,
      combined_vertical: 400,
      vertical_drop: 125,
    });
    expect(result.minimumElevation).toBe(100);
    expect(result.maximumElevation).toBe(700);
    expect(result.weightedMeans.bearing_alignment).toBeCloseTo(0.625);
  });

  it("does not coerce missing values to zero", () => {
    const result = calculateFilteredAggregates([
      skiArea({
        run_count: null,
        lift_count: null,
        combined_vertical: null,
        vertical_drop: null,
        min_elevation: null,
        max_elevation: null,
        bearing_alignment: null,
      }),
    ]);

    expect(result.sums).toMatchObject({
      run_count: null,
      lift_count: null,
      combined_vertical: null,
      vertical_drop: null,
    });
    expect(result.minimumElevation).toBeNull();
    expect(result.maximumElevation).toBeNull();
    expect(result.weightedMeans.bearing_alignment).toBeNull();
  });

  it("returns defined empty-result aggregates", () => {
    expect(calculateFilteredAggregates([])).toEqual({
      rowCount: 0,
      distinctCounts: {
        ski_area_name: 0,
        country: 0,
        region: 0,
        locality: 0,
      },
      sums: {
        run_count: null,
        lift_count: null,
        combined_vertical: null,
        vertical_drop: null,
      },
      minimumElevation: null,
      maximumElevation: null,
      weightedMeans: {
        bearing_alignment: null,
        poleward_affinity: null,
        eastward_affinity: null,
        run_proportion_4_north: null,
        run_proportion_4_east: null,
        run_proportion_4_south: null,
        run_proportion_4_west: null,
        run_proportion_2_north: null,
      },
    });
  });
});
