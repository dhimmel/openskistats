import { describe, expect, it } from "vitest";

import { matchesNumericFilter } from "../src/filters";
import {
  binLast,
  boundsFromBins,
  buildHistogram,
  describeBin,
  describeBounds,
  overlapsBin,
  parseBound,
  restricts,
  roundTo,
  UNBOUNDED,
  withBound,
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
      const last = clipped!.bins.length - 1;
      const bounds = boundsFromBins(clipped!, last, last);
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

describe("boundsFromBins", () => {
  // Bars of width 5 from 0 to 100.
  const histogram = histogramOf([0, 100]);
  const lastIndex = histogram.bins.length - 1;

  it("orders the bars a backwards drag supplies", () => {
    expect(boundsFromBins(histogram, 12, 4)).toEqual({ lower: 20, upper: 65 });
  });

  it("keeps an interior upper edge inclusive, as a typed bound is", () => {
    const values = [3, 3, 7, 12, 48, 90];
    const brushed = histogramOf(values);
    const index = brushed.bins.findIndex(
      (candidate, position) => candidate.count > 0 && position > 0,
    );
    const bin = brushed.bins[index];
    const bounds = boundsFromBins(brushed, index, index);
    expect(values.filter((value) => matchesNumericFilter(value, bounds))).toEqual(
      values.filter((value) => value >= bin.start && value <= bin.end),
    );
  });

  it.each([
    { bars: [4, lastIndex], purpose: "the upper end", lower: 20, upper: Number.POSITIVE_INFINITY },
    { bars: [0, 11], purpose: "the lower end", lower: Number.NEGATIVE_INFINITY, upper: 60 },
    { bars: [0, lastIndex], purpose: "both ends", lower: Number.NEGATIVE_INFINITY, upper: Number.POSITIVE_INFINITY },
  ])("releases a bound that reaches $purpose of the axis", ({ bars, lower, upper }) => {
    expect(boundsFromBins(histogram, bars[0], bars[1])).toEqual({ lower, upper });
  });

  describe("integer columns", () => {
    it("snaps an interior upper edge to the last whole number in the bar", () => {
      expect(boundsFromBins(histogram, 4, 11, true)).toEqual({ lower: 20, upper: 59 });
    });

    it("selects a single-width bar as one value", () => {
      const narrow = histogramOf([0, 1, 2, 3, 4, 5]);
      expect(narrow.step).toBe(1);
      expect(boundsFromBins(narrow, 3, 3, true)).toEqual({ lower: 3, upper: 3 });
    });

    it("still releases an edge at the end of the axis", () => {
      expect(boundsFromBins(histogram, 4, lastIndex, true).upper).toBe(
        Number.POSITIVE_INFINITY,
      );
    });
  });
});

describe("binLast", () => {
  const narrow = histogramOf([0, 1, 2, 3, 4, 5]);

  it("names an interior integer bar by its last whole number", () => {
    expect(binLast(narrow, 3, true)).toBe(3);
  });

  it("keeps the closed last bar's end, which holds the maximum", () => {
    expect(binLast(narrow, narrow.bins.length - 1, true)).toBe(5);
  });

  it("keeps a continuous bar's edge", () => {
    expect(binLast(narrow, 3)).toBe(4);
  });
});

describe("overlapsBin", () => {
  // Bars of width 1 from 0 to 5; the last, `[4, 5]`, is closed.
  const narrow = histogramOf([0, 1, 2, 3, 4, 5]);
  const last = narrow.bins.length - 1;

  it.each([
    { bounds: UNBOUNDED, index: 3, expected: true, purpose: "no bounds" },
    { bounds: { lower: 3, upper: 4 }, index: 3, expected: true, purpose: "bounds matching the bar's edges" },
    { bounds: { lower: 3.5, upper: 10 }, index: 3, expected: true, purpose: "bounds starting inside the bar" },
    { bounds: { lower: 4, upper: 10 }, index: 3, expected: false, purpose: "bounds starting at the bar's exclusive edge" },
    { bounds: { lower: 0, upper: 3 }, index: 3, expected: false, purpose: "continuous bounds ending at the bar's start" },
    { bounds: { lower: 5, upper: 10 }, index: last, expected: true, purpose: "bounds starting at the closed last bar's end" },
    { bounds: { lower: 5.5, upper: 10 }, index: last, expected: false, purpose: "bounds past the closed last bar" },
  ])("lights a bar for $purpose: $expected", ({ bounds, index, expected }) => {
    expect(overlapsBin(bounds, narrow, index)).toBe(expected);
  });

  it("lights an integer bar selected as its single value", () => {
    expect(overlapsBin({ lower: 3, upper: 3 }, narrow, 3, true)).toBe(true);
    expect(overlapsBin({ lower: 3, upper: 3 }, narrow, 4, true)).toBe(false);
    expect(overlapsBin({ lower: 3, upper: 3 }, narrow, 2, true)).toBe(false);
  });

  it("lights the closed last bar for the maximum it holds", () => {
    expect(overlapsBin({ lower: 5, upper: 5 }, narrow, last, true)).toBe(true);
  });

  it("lights an end bar for typed bounds beyond a clipped axis", () => {
    const ordinary = Array.from({ length: 400 }, (_unused, index) => index % 40);
    const above = histogramOf([...ordinary, 70_000]);
    expect(overlapsBin({ lower: 1_000, upper: Number.POSITIVE_INFINITY }, above, above.bins.length - 1)).toBe(true);
    const below = histogramOf([...ordinary, -70_000]);
    expect(overlapsBin({ lower: Number.NEGATIVE_INFINITY, upper: -1_000 }, below, 0)).toBe(true);
    expect(overlapsBin({ lower: Number.NEGATIVE_INFINITY, upper: -1_000 }, below, 1)).toBe(false);
  });
});

describe("describeBin", () => {
  const format = (value: number) => String(value);
  const narrow = histogramOf([0, 1, 2, 3, 4, 5]);
  const last = narrow.bins.length - 1;

  it.each([
    { index: 1, integer: false, expected: "1 to 2", purpose: "a continuous bar" },
    { index: 1, integer: true, expected: "1", purpose: "an integer bar of one value" },
    { index: last, integer: true, expected: "4 to 5", purpose: "the closed last integer bar" },
  ])("names $purpose", ({ index, integer, expected }) => {
    expect(describeBin(narrow, index, integer, format)).toBe(expected);
  });

  it("names the whole numbers a wide integer bar holds", () => {
    const wide = histogramOf([0, 100]);
    expect(describeBin(wide, 1, true, format)).toBe("5 to 9");
  });

  it("names the outliers an end bar absorbs", () => {
    const ordinary = Array.from({ length: 400 }, (_unused, index) => index % 40);
    const above = histogramOf([...ordinary, 70_000]);
    expect(describeBin(above, above.bins.length - 1, true, format)).toMatch(/ and above$/);
    const below = histogramOf([...ordinary, -70_000]);
    expect(describeBin(below, 0, true, format)).toMatch(/ and below$/);
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
    {
      edge: "lower",
      text: "Infinity",
      expected: null,
      purpose: "a bound that is not finite",
    },
  ] as const)("reads $purpose", ({ edge, text, expected }) => {
    expect(parseBound(edge, text)).toBe(expected);
  });
});

describe("withBound", () => {
  const bounds = { lower: 20, upper: 60 };

  it.each([
    { edge: "lower", value: 30, expected: { lower: 30, upper: 60 }, purpose: "a minimum inside the range" },
    { edge: "upper", value: 50, expected: { lower: 20, upper: 50 }, purpose: "a maximum inside the range" },
    { edge: "lower", value: Number.NEGATIVE_INFINITY, expected: { lower: Number.NEGATIVE_INFINITY, upper: 60 }, purpose: "an emptied minimum" },
    { edge: "lower", value: 80, expected: { lower: 80, upper: Number.POSITIVE_INFINITY }, purpose: "a minimum past the maximum, which releases it" },
    { edge: "upper", value: 10, expected: { lower: Number.NEGATIVE_INFINITY, upper: 10 }, purpose: "a maximum below the minimum, which releases it" },
    { edge: "upper", value: 20, expected: { lower: 20, upper: 20 }, purpose: "a maximum equal to the minimum" },
  ] as const)("applies $purpose", ({ edge, value, expected }) => {
    expect(withBound(bounds, edge, value)).toEqual(expected);
  });
});
