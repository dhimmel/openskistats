import { describe, expect, it } from "vitest";

import { matchesNumericFilter } from "../src/filters";
import {
  boundsFromEdges,
  buildHistogram,
  describeBounds,
  parseBound,
  restricts,
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
      const bounds = boundsFromEdges(last.start, last.end, clipped!);
      expect(matchesNumericFilter(70_000, bounds)).toBe(true);
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

describe("boundsFromEdges", () => {
  const histogram = histogramOf([0, 100]);

  it("orders the edges a backwards drag supplies", () => {
    const bounds = boundsFromEdges(60, 20, histogram);
    expect([bounds.lower, bounds.upper]).toEqual([20, 60]);
  });

  it("keeps an interior upper edge inclusive, as a typed bound is", () => {
    const values = [3, 3, 7, 12, 48, 90];
    const brushed = histogramOf(values);
    const bin = brushed.bins.find(
      (candidate) => candidate.count > 0 && candidate.start > brushed.start,
    );
    const bounds = boundsFromEdges(bin!.start, bin!.end, brushed);
    expect(values.filter((value) => matchesNumericFilter(value, bounds))).toEqual(
      values.filter((value) => value >= bin!.start && value <= bin!.end),
    );
  });

  it.each([
    { edges: [20, 100], purpose: "the upper end", lower: 20, upper: Number.POSITIVE_INFINITY },
    { edges: [0, 60], purpose: "the lower end", lower: Number.NEGATIVE_INFINITY, upper: 60 },
    { edges: [0, 100], purpose: "both ends", lower: Number.NEGATIVE_INFINITY, upper: Number.POSITIVE_INFINITY },
  ])("releases a bound that reaches $purpose of the axis", ({ edges, lower, upper }) => {
    expect(boundsFromEdges(edges[0], edges[1], histogram)).toEqual({ lower, upper });
  });
});

describe("restricts", () => {
  it("is false for bounds that cover everything, so the filter clears", () => {
    expect(restricts(UNBOUNDED)).toBe(false);
  });

  it.each([
    { bounds: { lower: 20, upper: Number.POSITIVE_INFINITY }, purpose: "a lower bound" },
    { bounds: { lower: Number.NEGATIVE_INFINITY, upper: 60 }, purpose: "an upper bound" },
  ])("is true for $purpose", ({ bounds }) => {
    expect(restricts(bounds)).toBe(true);
  });
});

describe("describeBounds", () => {
  const format = (value: number) => String(value);

  it.each([
    { bounds: UNBOUNDED, expected: "Any", purpose: "no bounds" },
    { bounds: { lower: 3, upper: 10 }, expected: "3–10", purpose: "both bounds" },
    {
      bounds: { lower: 3, upper: Number.POSITIVE_INFINITY },
      expected: "≥ 3",
      purpose: "a lower bound",
    },
    {
      bounds: { lower: Number.NEGATIVE_INFINITY, upper: 10 },
      expected: "≤ 10",
      purpose: "an upper bound",
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
