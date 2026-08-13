import { describe, expect, it } from "vitest";

import {
  readSkiAreaDocument,
  SkiAreaContractError,
  type SkiAreaDocument,
} from "../src/types";

function document(overrides: Record<string, unknown> = {}): SkiAreaDocument {
  return {
    schema_version: "1.0",
    data_updated_at: "2026-01-01T00:00:00Z",
    license: "ODbL-1.0",
    attribution: "Attribution",
    sources: [],
    record_count: 1,
    record_schema: { properties: {} },
    ski_areas: [
      {
        ski_area_id: "example",
        ski_area_name: "Example",
      } as SkiAreaDocument["ski_areas"][number],
    ],
    ...overrides,
  };
}

describe("readSkiAreaDocument", () => {
  it("accepts the supported contract", () => {
    const value = document();
    expect(readSkiAreaDocument(value)).toBe(value);
  });

  it.each([
    {
      overrides: { schema_version: "2.0" },
      message: "requires \"1.0\"",
      purpose: "unsupported schema version",
    },
    {
      overrides: { record_count: 2 },
      message: "reports 2 records but contains 1",
      purpose: "inconsistent record count",
    },
    {
      overrides: { record_schema: {} },
      message: "record_schema.properties",
      purpose: "missing property schema",
    },
    {
      overrides: { ski_areas: [{}] },
      message: "ski_area_id or ski_area_name",
      purpose: "missing record identity",
    },
  ])("rejects $purpose", ({ overrides, message }) => {
    expect(() => readSkiAreaDocument(document(overrides))).toThrowError(
      new RegExp(message),
    );
  });

  it("uses a dedicated error type", () => {
    expect(() => readSkiAreaDocument(null)).toThrow(SkiAreaContractError);
  });
});
