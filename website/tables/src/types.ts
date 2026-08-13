export const SUPPORTED_SCHEMA_VERSION = "1.0" as const;

export interface JsonSchemaProperty {
  description?: string;
  [key: string]: unknown;
}

export interface TableRecordSchema {
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
  record_schema: TableRecordSchema;
  ski_areas: SkiAreaSummary[];
}

/** Retained name for the ski-area table's schema type. */
export type SkiAreaRecordSchema = TableRecordSchema;

export interface LiftSummary {
  lift_id: string;
  lift_name: string;
  lift_type: string;
  lift_status: string;
  lift_access: string | null;
  lift_oneway: boolean | null;
  lift_occupancy: number | null;
  lift_capacity: number | null;
  lift_duration: number | null;
  lift_detachable: boolean | null;
  lift_bubble: boolean | null;
  lift_heating: boolean | null;
  ski_area_ids: string[];
  ski_area_names: (string | null)[];
  country: string | null;
  country_code: string | null;
  country_subdiv_code: string | null;
  region: string | null;
  locality: string | null;
  latitude: number | null;
  longitude: number | null;
  lift_websites: string[];
  lift_sources: string[];
  wikidata_id: string | null;
  inclined_length: number | null;
  vertical_rise: number | null;
  min_elevation: number | null;
  max_elevation: number | null;
}

export interface LiftDocument {
  schema_version: typeof SUPPORTED_SCHEMA_VERSION;
  data_updated_at: string;
  license: "ODbL-1.0";
  attribution: string;
  sources: unknown[];
  record_count: number;
  record_schema: TableRecordSchema;
  lifts: LiftSummary[];
}

export class TableContractError extends Error {
  override name = "TableContractError";
}

/** Retained name for the ski-area table's error type. */
export const SkiAreaContractError = TableContractError;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

interface DocumentSpec {
  /** Envelope key holding the records, such as `ski_areas`. */
  recordsKey: string;
  /** Human-readable noun used in error messages. */
  label: string;
  /** Record fields that must be present as strings. */
  requiredStringFields: readonly string[];
}

/**
 * Perform inexpensive runtime checks on the published document envelope.
 *
 * Pydantic validates the complete record schema when publishing the JSON.
 * The browser only checks the compatibility boundary and fields needed to safely
 * construct the table.
 */
function readDocument<TDocument>(value: unknown, spec: DocumentSpec): TDocument {
  const { label, recordsKey, requiredStringFields } = spec;
  if (!isObject(value)) {
    throw new TableContractError(`The ${label} data response is not a JSON object.`);
  }

  if (value.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    const received =
      typeof value.schema_version === "string"
        ? `\"${value.schema_version}\"`
        : "missing or invalid";
    throw new TableContractError(
      `Unsupported ${label} data schema version ${received}; ` +
        `this table requires \"${SUPPORTED_SCHEMA_VERSION}\". ` +
        "The data export and website frontend must be deployed together.",
    );
  }

  if (
    typeof value.record_count !== "number" ||
    !Number.isSafeInteger(value.record_count) ||
    value.record_count < 0
  ) {
    throw new TableContractError(`The ${label} data has an invalid record_count.`);
  }
  const records = value[recordsKey];
  if (!Array.isArray(records)) {
    throw new TableContractError(
      `The ${label} data is missing its ${recordsKey} array.`,
    );
  }
  if (value.record_count !== records.length) {
    throw new TableContractError(
      `The ${label} data reports ${value.record_count} records but contains ` +
        `${records.length}.`,
    );
  }

  if (!isObject(value.record_schema) || !isObject(value.record_schema.properties)) {
    throw new TableContractError(
      `The ${label} data is missing record_schema.properties.`,
    );
  }

  records.forEach((record, index) => {
    if (
      !isObject(record) ||
      requiredStringFields.some((field) => typeof record[field] !== "string")
    ) {
      throw new TableContractError(
        `${label} record ${index + 1} is missing a string ` +
          `${requiredStringFields.join(" or ")}.`,
      );
    }
  });

  return value as TDocument;
}

export function readSkiAreaDocument(value: unknown): SkiAreaDocument {
  return readDocument<SkiAreaDocument>(value, {
    label: "ski-area",
    recordsKey: "ski_areas",
    requiredStringFields: ["ski_area_id", "ski_area_name"],
  });
}

export function readLiftDocument(value: unknown): LiftDocument {
  return readDocument<LiftDocument>(value, {
    label: "lift",
    recordsKey: "lifts",
    requiredStringFields: ["lift_id", "lift_name"],
  });
}
