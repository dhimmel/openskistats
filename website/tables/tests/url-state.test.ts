import { afterEach, describe, expect, it, vi } from "vitest";

import {
  BOOLEAN_URL_VALUE,
  readTableState,
  replaceSearch,
  searchStore,
  type TableUrlSpec,
  type TableUrlState,
  urlColumnsFrom,
  writeTableState,
} from "../src/url-state";

const atLeast = (lower: number) => ({ lower, upper: Number.POSITIVE_INFINITY });
const atMost = (upper: number) => ({ lower: Number.NEGATIVE_INFINITY, upper });

const CODES: Record<string, string> = {
  Austria: "AT",
  France: "FR",
  Switzerland: "CH",
};
const NAMES = Object.fromEntries(
  Object.entries(CODES).map(([name, code]) => [code, name]),
);

const spec: TableUrlSpec = {
  columns: [
    { id: "ski_area_name", variant: "faceted" },
    {
      id: "country",
      urlValue: {
        decode: (text) => NAMES[text.toUpperCase()],
        encode: (value) => CODES[String(value)] ?? String(value),
      },
      variant: "faceted",
    },
    { id: "lift_detachable", urlValue: BOOLEAN_URL_VALUE, variant: "faceted" },
    { id: "run_count", variant: "range" },
    { id: "latitude", variant: "range" },
    { id: "poleward_affinity", variant: "range" },
  ],
  defaults: {
    columnFilters: [{ id: "run_count", value: atLeast(3) }],
    sorting: [{ desc: true, id: "run_count" }],
  },
};

/** The default state, as a fresh table shows it. */
const defaults = (): TableUrlState => ({
  columnFilters: [{ id: "run_count", value: atLeast(3) }],
  sorting: [{ desc: true, id: "run_count" }],
});

function filterOf(search: string, id: string): unknown {
  return readTableState(search, spec).columnFilters.find((filter) => filter.id === id)
    ?.value;
}

describe("readTableState", () => {
  it("applies the defaults to a bare URL", () => {
    expect(readTableState("", spec)).toEqual(defaults());
  });

  it("clears a default filter given an empty value", () => {
    expect(readTableState("?run_count=", spec).columnFilters).toEqual([]);
  });

  it.each([
    { text: "3..", expected: atLeast(3), purpose: "a lower bound" },
    { text: "..10", expected: atMost(10), purpose: "an upper bound" },
    { text: "3..10", expected: { lower: 3, upper: 10 }, purpose: "both bounds" },
    { text: "-45..-30", expected: { lower: -45, upper: -30 }, purpose: "negative bounds" },
    { text: "..0", expected: atMost(0), purpose: "the southern hemisphere" },
    { text: ".5..1.5", expected: { lower: 0.5, upper: 1.5 }, purpose: "decimal bounds" },
    { text: "60..", expected: atLeast(60), purpose: "a percent column in displayed units" },
    { text: "1e-7..", expected: atLeast(1e-7), purpose: "a bound written in exponent notation" },
    { text: "-1.5E2..", expected: atLeast(-150), purpose: "an upper-case exponent" },
  ])("reads $purpose", ({ text, expected }) => {
    expect(filterOf(`?latitude=${text}`, "latitude")).toEqual(expected);
  });

  it.each([
    { text: "abc", purpose: "text" },
    { text: "3", purpose: "a bare number" },
    { text: "10..3", purpose: "a lower bound above the upper one" },
    { text: "..", purpose: "no bounds at all" },
    { text: "Infinity..", purpose: "a non-finite bound" },
    { text: "1e400..", purpose: "an exponent that overflows" },
    { text: `1${"0".repeat(400)}..`, purpose: "a decimal that overflows" },
    { text: "0x10..", purpose: "a hexadecimal bound" },
    { text: "3..ten", purpose: "one unreadable bound" },
  ])("falls back to the default for $purpose", ({ text }) => {
    expect(filterOf(`?run_count=${text}`, "run_count")).toEqual(atLeast(3));
  });

  it("reads repeated keys as a selection", () => {
    expect(filterOf("?country=AT&country=CH", "country")).toEqual([
      "Austria",
      "Switzerland",
    ]);
  });

  it("reads a country code in either case", () => {
    expect(filterOf("?country=at", "country")).toEqual(["Austria"]);
  });

  it("ignores an unknown country code", () => {
    expect(filterOf("?country=ZZ&country=AT", "country")).toEqual(["Austria"]);
    expect(filterOf("?country=ZZ", "country")).toBeUndefined();
  });

  it("deduplicates repeated values", () => {
    expect(filterOf("?country=AT&country=at", "country")).toEqual(["Austria"]);
  });

  it("drops an empty value beside real ones", () => {
    expect(filterOf("?ski_area_name=&ski_area_name=Alta", "ski_area_name")).toEqual([
      "Alta",
    ]);
  });

  it("reads the blank option", () => {
    expect(filterOf("?ski_area_name=null", "ski_area_name")).toEqual([null]);
  });

  it("reads booleans through their codec", () => {
    expect(filterOf("?lift_detachable=true&lift_detachable=null", "lift_detachable")).toEqual(
      [true, null],
    );
    expect(filterOf("?lift_detachable=maybe", "lift_detachable")).toBeUndefined();
  });

  it.each(["constructor", "toString", "__proto__", "TRUE"])(
    "does not read %s as a boolean",
    (text) => {
      expect(filterOf(`?lift_detachable=${text}`, "lift_detachable")).toBeUndefined();
    },
  );

  it("round-trips a bound typed with more precision than the axis", () => {
    const state = {
      columnFilters: [{ id: "latitude", value: atLeast(0.0000001) }],
      sorting: defaults().sorting,
    };
    const search = writeTableState("", { ...defaults(), ...state, columnFilters: [...defaults().columnFilters, ...state.columnFilters] }, spec);
    expect(search).toBe("?latitude=1e-7..");
    expect(filterOf(search, "latitude")).toEqual(atLeast(1e-7));
  });

  it.each([
    { text: null, expected: defaults().sorting, purpose: "no parameter" },
    { text: "", expected: [], purpose: "an empty parameter" },
    { text: "latitude", expected: [{ desc: false, id: "latitude" }], purpose: "ascending" },
    { text: "-latitude", expected: [{ desc: true, id: "latitude" }], purpose: "descending" },
    {
      text: "-country,latitude",
      expected: [
        { desc: true, id: "country" },
        { desc: false, id: "latitude" },
      ],
      purpose: "several sorts in priority order",
    },
    { text: "-bogus", expected: defaults().sorting, purpose: "an unknown column" },
    {
      text: "-bogus,latitude",
      expected: [{ desc: false, id: "latitude" }],
      purpose: "an unknown column beside a known one",
    },
  ])("reads sorting from $purpose", ({ text, expected }) => {
    const search = text === null ? "" : `?sort=${text}`;
    expect(readTableState(search, spec).sorting).toEqual(expected);
  });

  it("ignores parameters the table does not own", () => {
    expect(readTableState("?foo=bar&sort=", spec)).toEqual({
      columnFilters: defaults().columnFilters,
      sorting: [],
    });
  });
});

describe("writeTableState", () => {
  it("leaves the URL bare for the default state", () => {
    expect(writeTableState("", defaults(), spec)).toBe("");
  });

  it("writes a cleared default as an empty value", () => {
    expect(writeTableState("", { columnFilters: [], sorting: defaults().sorting }, spec)).toBe(
      "?run_count=",
    );
  });

  it("writes facet values in sorted order, as codes for countries", () => {
    const state = {
      ...defaults(),
      columnFilters: [
        ...defaults().columnFilters,
        { id: "country", value: ["Switzerland", "Austria"] },
      ],
    };
    expect(writeTableState("", state, spec)).toBe("?country=AT&country=CH");
  });

  it("preserves parameters the table does not own", () => {
    const state = {
      ...defaults(),
      columnFilters: [...defaults().columnFilters, { id: "country", value: ["Austria"] }],
    };
    expect(writeTableState("?foo=bar&country=FR", state, spec)).toBe("?foo=bar&country=AT");
  });

  it("removes a table parameter that no longer applies", () => {
    expect(writeTableState("?country=FR&sort=latitude", defaults(), spec)).toBe("");
  });

  it.each([
    { sorting: [{ desc: false, id: "latitude" }], expected: "?sort=latitude" },
    { sorting: [], expected: "?sort=" },
  ])("writes sorting $expected", ({ sorting, expected }) => {
    expect(writeTableState("", { ...defaults(), sorting }, spec)).toBe(expected);
  });

  it.each([
    {
      purpose: "names with punctuation, spaces, and accents",
      state: {
        columnFilters: [
          {
            id: "ski_area_name",
            value: [
              "100% Ski & Board",
              "Alpe d'Huez, Grand Domaine",
              "Val d'Isère",
              "Zermatt/Cervinia+",
            ],
          },
        ],
        sorting: [],
      },
    },
    {
      purpose: "the blank option beside a value",
      state: {
        columnFilters: [{ id: "country", value: ["France", null] }],
        sorting: defaults().sorting,
      },
    },
    {
      purpose: "booleans",
      state: {
        columnFilters: [{ id: "lift_detachable", value: [false, true] }],
        sorting: defaults().sorting,
      },
    },
    {
      purpose: "negative and decimal ranges with several sorts",
      state: {
        columnFilters: [
          { id: "run_count", value: atLeast(3) },
          { id: "latitude", value: { lower: -45.5, upper: -30 } },
          { id: "poleward_affinity", value: atLeast(60) },
        ],
        sorting: [
          { desc: true, id: "country" },
          { desc: false, id: "latitude" },
        ],
      },
    },
  ])("round-trips $purpose", ({ state }) => {
    // Filters come back in column order and facet values in written order.
    expect(readTableState(writeTableState("", state, spec), spec)).toEqual(state);
  });

  it("writes a URL back unchanged", () => {
    const search = "?foo=bar&country=AT&country=CH&run_count=&sort=latitude";
    expect(writeTableState(search, readTableState(search, spec), spec)).toBe(search);
  });
});

describe("urlColumnsFrom", () => {
  it("flattens groups to the filterable leaves, with their codecs", () => {
    const columns = urlColumnsFrom([
      {
        columns: [
          { id: "ski_area_id" },
          { id: "country", meta: { filterVariant: "faceted", urlValue: BOOLEAN_URL_VALUE } },
        ],
        header: "Location",
      },
      { id: "run_count", meta: { filterVariant: "range" } },
    ]);
    expect(columns).toEqual([
      { id: "country", urlValue: BOOLEAN_URL_VALUE, variant: "faceted" },
      { id: "run_count", urlValue: undefined, variant: "range" },
    ]);
  });
});

/** A history the tests can drive, standing in for the browser's. */
function fakeWindow(href: string) {
  const entries = [href];
  let index = 0;
  const handlers = new Set<() => void>();
  const notify = () => handlers.forEach((handler) => handler());
  const replaced: unknown[] = [];
  return {
    get location() {
      return new URL(entries[index]);
    },
    history: {
      back: () => {
        index -= 1;
        notify();
      },
      forward: () => {
        index += 1;
        notify();
      },
      pushState: (_state: unknown, _title: string, url: string) => {
        entries.splice(index + 1);
        entries.push(new URL(url, entries[index]).href);
        index += 1;
      },
      replaceState: (state: unknown, _title: string, url: string) => {
        replaced.push(state);
        entries[index] = new URL(url, entries[index]).href;
      },
      state: { scroll: 0 },
    },
    addEventListener: (_type: string, handler: () => void) => handlers.add(handler),
    removeEventListener: (_type: string, handler: () => void) => handlers.delete(handler),
    replaced,
  };
}

describe("searchStore", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("follows Back and Forward across an anchor navigation", () => {
    const window = fakeWindow("https://openskistats.org/ski-areas/");
    vi.stubGlobal("window", window);
    const seen: string[] = [];
    searchStore.subscribe(() => seen.push(searchStore.getSnapshot()));

    replaceSearch("?country=AT");
    window.history.pushState(null, "", "#table-usage");
    replaceSearch("?country=CH");
    expect(window.location.href).toBe(
      "https://openskistats.org/ski-areas/?country=CH#table-usage",
    );

    window.history.back();
    expect(searchStore.getSnapshot()).toBe("?country=AT");
    window.history.forward();
    expect(searchStore.getSnapshot()).toBe("?country=CH");
    expect(seen).toEqual(["?country=AT", "?country=CH", "?country=AT", "?country=CH"]);
  });

  it("keeps the history entry's state and skips a write that changes nothing", () => {
    const window = fakeWindow("https://openskistats.org/lifts/?lift_type=gondola");
    vi.stubGlobal("window", window);

    replaceSearch("?lift_type=gondola");
    replaceSearch("");
    expect(window.location.href).toBe("https://openskistats.org/lifts/");
    expect(window.replaced).toEqual([{ scroll: 0 }]);
  });
});
