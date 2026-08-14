import { describe, expect, it } from "vitest";

import { matchesNumericFilter } from "../src/filters";
import {
  boundsFromEdges,
  boundsFromFilter,
  buildHistogram,
  describeBounds,
  formatRangeFilter,
  parseBound,
  roundTo,
  UNBOUNDED,
} from "../src/range";

/** A histogram is only ever built from values, so tests may assume one. */
function histogramOf(values: readonly (number | null)[]) {
  const histogram = buildHistogram(values);
  if (histogram === null) {
    throw new Error("expected a histogram");
  }
  return histogram;
}

describe("buildHistogram", () => {
  it("returns null without a numeric value", () => {
    expect(buildHistogram([null, undefined, Number.NaN])).toBeNull();
  });

  it("counts missing values apart from binned ones", () => {
    const histogram = histogramOf([1, 2, null, 3, null]);
    expect(histogram.missingCount).toBe(2);
    expect(histogram.valueCount).toBe(3);
  });

  it.each([
    { values: [0, 100], step: 5, purpose: "a round span" },
    { values: [0, 1], step: 1, purpose: "integers narrower than the bin target" },
    { values: [0, 4800], step: 200, purpose: "elevations in metres" },
    { values: [0, 0.5, 1.2], step: 0.05, purpose: "fractional values" },
  ])("chooses a legible bin width for $purpose", ({ values, step }) => {
    expect(histogramOf(values).step).toBe(step);
  });

  it("covers every value, tallying each into exactly one bin", () => {
    const values = [3, 3, 7, 12, 48, 90];
    const histogram = histogramOf(values);
    expect(histogram.start).toBeLessThanOrEqual(3);
    expect(histogram.end).toBeGreaterThanOrEqual(90);
    expect(histogram.bins.reduce((total, bin) => total + bin.count, 0)).toBe(
      values.length,
    );
    expect(histogram.maxCount).toBe(2);
  });

  it("gives a column with one distinct value a single populated bin", () => {
    const histogram = histogramOf([7, 7, 7]);
    expect(histogram.end).toBeGreaterThan(histogram.start);
    expect(histogram.bins.filter((bin) => bin.count > 0)).toHaveLength(1);
  });

  describe("outliers", () => {
    const ordinary = Array.from({ length: 400 }, (_unused, index) => index % 40);
    const clipped = buildHistogram([...ordinary, 70_000]);

    it("stops the axis short of a value that would flatten the bars", () => {
      expect(clipped!.end).toBeLessThan(100);
      expect(clipped!.maximum).toBe(70_000);
    });

    it("still counts the outlier, in the end bar", () => {
      expect(clipped!.bins.reduce((total, bin) => total + bin.count, 0)).toBe(
        ordinary.length + 1,
      );
    });

    it("selects the outlier along with the end bar it sits in", () => {
      const last = clipped!.bins[clipped!.bins.length - 1];
      const filter = formatRangeFilter(
        boundsFromEdges(last.start, last.end, clipped!),
        clipped!,
      );
      expect(matchesNumericFilter(70_000, filter)).toBe(true);
    });

    it("bins over the full extent when no value dominates it", () => {
      const histogram = histogramOf(ordinary);
      expect(histogram.start).toBe(histogram.minimum);
      expect(histogram.end).toBeGreaterThanOrEqual(histogram.maximum);
    });
  });

  it("keeps bin edges free of floating-point drift", () => {
    for (const bin of histogramOf([0, 0.5, 1.2]).bins) {
      expect(bin.start).toBe(roundTo(bin.start, 4));
    }
  });
});

describe("formatRangeFilter", () => {
  const histogram = histogramOf([0, 100]);

  it("clears the filter when the bounds cover the distribution", () => {
    expect(formatRangeFilter(UNBOUNDED, histogram)).toBeUndefined();
  });

  it.each([
    {
      bounds: { lower: 20, lowerInclusive: true, upper: 60, upperInclusive: true },
      expected: "[20, 60]",
      purpose: "a closed range",
    },
    {
      bounds: {
        lower: 20,
        lowerInclusive: true,
        upper: Number.POSITIVE_INFINITY,
        upperInclusive: true,
      },
      expected: "[20, ]",
      purpose: "a lower bound only",
    },
    {
      bounds: {
        lower: Number.NEGATIVE_INFINITY,
        lowerInclusive: true,
        upper: 60,
        upperInclusive: false,
      },
      expected: "[, 60)",
      purpose: "an exclusive upper bound only",
    },
  ])("writes $purpose", ({ bounds, expected }) => {
    expect(formatRangeFilter(bounds, histogram)).toBe(expected);
  });

  it("round-trips through the filter it writes", () => {
    const values = [3, 3, 7, 12, 48, 90];
    const brushed = histogramOf(values);
    // An interior bin, so that both bounds survive into the filter text.
    const bin = brushed.bins.find(
      (candidate) => candidate.count > 0 && candidate.start > brushed.start,
    );
    const filter = formatRangeFilter(
      boundsFromEdges(bin!.start, bin!.end, brushed),
      brushed,
    );
    expect(values.filter((value) => matchesNumericFilter(value, filter))).toEqual(
      values.filter((value) => value >= bin!.start && value < bin!.end),
    );
    expect(boundsFromFilter(filter).lower).toBe(bin!.start);
  });

  it("excludes rows without a value once bounded", () => {
    const filter = formatRangeFilter(
      { lower: 20, lowerInclusive: true, upper: 60, upperInclusive: true },
      histogram,
    );
    expect(matchesNumericFilter(null, filter)).toBe(false);
  });
});

describe("boundsFromEdges", () => {
  const histogram = histogramOf([0, 100]);

  it("orders the edges a backwards drag supplies", () => {
    const bounds = boundsFromEdges(60, 20, histogram);
    expect([bounds.lower, bounds.upper]).toEqual([20, 60]);
  });

  it("leaves an interior upper edge exclusive so adjacent bins do not overlap", () => {
    expect(boundsFromEdges(20, 60, histogram).upperInclusive).toBe(false);
  });

  it("closes the range at the end of the distribution", () => {
    expect(boundsFromEdges(20, histogram.end, histogram).upperInclusive).toBe(true);
  });
});

describe("boundsFromFilter", () => {
  it.each([
    { filter: undefined, purpose: "an unset filter" },
    { filter: ["a", "b"], purpose: "a value picker's selection" },
    { filter: "nonsense", purpose: "text that is not a range" },
  ])("falls back to unbounded for $purpose", ({ filter }) => {
    expect(boundsFromFilter(filter)).toEqual(UNBOUNDED);
  });

  it("reads a threshold typed into the filter box", () => {
    const bounds = boundsFromFilter("3");
    expect(bounds.lower).toBe(3);
    expect(bounds.upper).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("describeBounds", () => {
  const format = (value: number) => String(value);

  it.each([
    { bounds: UNBOUNDED, expected: "Any", purpose: "no bounds" },
    {
      bounds: { lower: 3, lowerInclusive: true, upper: 10, upperInclusive: false },
      expected: "3–10",
      purpose: "both bounds",
    },
    {
      bounds: {
        lower: 3,
        lowerInclusive: true,
        upper: Number.POSITIVE_INFINITY,
        upperInclusive: true,
      },
      expected: "≥ 3",
      purpose: "a lower bound",
    },
    {
      bounds: {
        lower: Number.NEGATIVE_INFINITY,
        lowerInclusive: true,
        upper: 10,
        upperInclusive: false,
      },
      expected: "< 10",
      purpose: "an exclusive upper bound",
    },
  ])("summarises $purpose", ({ bounds, expected }) => {
    expect(describeBounds(bounds, format)).toBe(expected);
  });
});

describe("parseBound", () => {
  it.each([
    { edge: "lower", text: "20", expected: 20, purpose: "a typed number" },
    { edge: "upper", text: " 1.5 ", expected: 1.5, purpose: "surrounding space" },
    { edge: "lower", text: "-40", expected: -40, purpose: "a negative bound" },
    {
      edge: "lower",
      text: "",
      expected: Number.NEGATIVE_INFINITY,
      purpose: "an emptied lower box, which releases the bound",
    },
    {
      edge: "upper",
      text: "",
      expected: Number.POSITIVE_INFINITY,
      purpose: "an emptied upper box, which releases the bound",
    },
    {
      edge: "lower",
      text: "1,200",
      expected: null,
      purpose: "a grouped number, which Number cannot read",
    },
  ] as const)("reads $purpose", ({ edge, text, expected }) => {
    expect(parseBound(edge, text)).toBe(expected);
  });
});
