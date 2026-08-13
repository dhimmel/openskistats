export const SUPPORTED_SCHEMA_VERSION = "1.0" as const;

export interface JsonSchemaProperty {
  description?: string;
  [key: string]: unknown;
}

export interface SkiAreaRecordSchema {
  properties: Record<string, JsonSchemaProperty>;
  [key: string]: unknown;
}

export interface SkiAreaSummary {
  ski_area_id: string;
  ski_area_name: string;
  osm_status: string | null;
  osm_run_convention: string;
  ski_area_uses: string[] | null;
  country: string | null;
  country_code: string | null;
  country_subdiv_code: string | null;
  region: string | null;
  locality: string | null;
  latitude: number | null;
  longitude: number | null;
  ski_area_websites: string[] | null;
  ski_area_sources: string[] | null;
  wikidata_id: string | null;
  run_count: number | null;
  lift_count: number | null;
  combined_vertical: number | null;
  combined_distance: number | null;
  vertical_drop: number | null;
  min_elevation: number | null;
  max_elevation: number | null;
  solar_irradiation_season: number | null;
  bearing_mean: number | null;
  bearing_alignment: number | null;
  poleward_affinity: number | null;
  eastward_affinity: number | null;
  run_proportion_4_north: number | null;
  run_proportion_4_east: number | null;
  run_proportion_4_south: number | null;
  run_proportion_4_west: number | null;
  run_proportion_2_north: number | null;
}

export interface SkiAreaDocument {
  schema_version: typeof SUPPORTED_SCHEMA_VERSION;
  data_updated_at: string;
  license: "ODbL-1.0";
  attribution: string;
  sources: unknown[];
  record_count: number;
  record_schema: SkiAreaRecordSchema;
  ski_areas: SkiAreaSummary[];
}

export class SkiAreaContractError extends Error {
  override name = "SkiAreaContractError";
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Perform inexpensive runtime checks on the published document envelope.
 *
 * Pydantic validates the complete record schema when publishing the JSON.
 * The browser only checks the compatibility boundary and fields needed to safely
 * construct the table.
 */
export function readSkiAreaDocument(value: unknown): SkiAreaDocument {
  if (!isObject(value)) {
    throw new SkiAreaContractError("The ski-area data response is not a JSON object.");
  }

  if (value.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    const received =
      typeof value.schema_version === "string"
        ? `\"${value.schema_version}\"`
        : "missing or invalid";
    throw new SkiAreaContractError(
      `Unsupported ski-area data schema version ${received}; ` +
        `this table requires \"${SUPPORTED_SCHEMA_VERSION}\". ` +
        "The data export and website frontend must be deployed together.",
    );
  }

  if (
    typeof value.record_count !== "number" ||
    !Number.isSafeInteger(value.record_count) ||
    value.record_count < 0
  ) {
    throw new SkiAreaContractError("The ski-area data has an invalid record_count.");
  }
  if (!Array.isArray(value.ski_areas)) {
    throw new SkiAreaContractError("The ski-area data is missing its ski_areas array.");
  }
  if (value.record_count !== value.ski_areas.length) {
    throw new SkiAreaContractError(
      `The ski-area data reports ${value.record_count} records but contains ` +
        `${value.ski_areas.length}.`,
    );
  }

  if (!isObject(value.record_schema) || !isObject(value.record_schema.properties)) {
    throw new SkiAreaContractError(
      "The ski-area data is missing record_schema.properties.",
    );
  }

  value.ski_areas.forEach((record, index) => {
    if (
      !isObject(record) ||
      typeof record.ski_area_id !== "string" ||
      typeof record.ski_area_name !== "string"
    ) {
      throw new SkiAreaContractError(
        `Ski-area record ${index + 1} is missing a string ski_area_id or ski_area_name.`,
      );
    }
  });

  return value as unknown as SkiAreaDocument;
}
