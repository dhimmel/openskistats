/**
 * Binning and bound arithmetic behind the numeric range filters.
 *
 * Kept apart from the components that draw them so the arithmetic can be
 * tested without a DOM. Bounds are inclusive at both ends, whether typed into
 * the boxes or brushed across the bars, so the two are one state.
 */
import type { NumericRange } from "./filters";

/** An unrestricted interval: the state of a column with no range filter. */
export const UNBOUNDED: NumericRange = {
  lower: Number.NEGATIVE_INFINITY,
  upper: Number.POSITIVE_INFINITY,
};

/** Bins to aim for; the nice-number rounding lands somewhat either side. */
const TARGET_BIN_COUNT = 24;

/** Backstop against a pathological outlier demanding thousands of bars. */
const MAXIMUM_BIN_COUNT = 200;

export interface HistogramBin {
  count: number;
  end: number;
  start: number;
}

export interface Histogram {
  bins: HistogramBin[];
  /** Right edge of the last bin, which may sit below `maximum`. */
  end: number;
  /** Largest value present, whether or not the bins reach it. */
  maximum: number;
  /** Tallest bin, which scales the drawn bars. */
  maxCount: number;
  /** Smallest value present, whether or not the bins reach it. */
  minimum: number;
  /** Rows without a value, which every range filter excludes. */
  missingCount: number;
  /** Decimal places that describe a bin edge exactly. */
  precision: number;
  /** Left edge of the first bin, which may sit above `minimum`. */
  start: number;
  step: number;
  /** Rows carrying a value, whatever the current bounds keep. */
  valueCount: number;
}

/** Round to `precision` decimals, clearing the drift of repeated addition. */
export function roundTo(value: number, precision: number): number {
  return Number(value.toFixed(precision));
}

/**
 * A bin width of 1, 2, or 5 times a power of ten, so that edges land on
 * numbers a reader recognises rather than on `span / 24`.
 */
function niceStep(span: number, integral: boolean): number {
  const rough = span / TARGET_BIN_COUNT;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  // Geometric thresholds, so the chosen width is the nearest of the three in
  // proportion rather than always the next one up: rounding up every time
  // halves the bar count whenever the ideal width falls just past a step.
  const factor =
    normalized >= Math.sqrt(50)
      ? 10
      : normalized >= Math.sqrt(10)
        ? 5
        : normalized >= Math.sqrt(2)
          ? 2
          : 1;
  // A power of ten times 1, 2, or 5 is already whole once it reaches 1, so
  // integer columns only need the sub-unit widths lifted.
  return integral ? Math.max(1, factor * magnitude) : factor * magnitude;
}

/** Share of the values a clipped domain may push into its end bins. */
const CLIP_QUANTILE = 0.005;

/**
 * The share of the full extent below which an outlier is judged to be
 * flattening the distribution rather than describing it.
 */
const CLIP_THRESHOLD = 0.5;

/**
 * The extent to bin over: the full one, unless a lone extreme value would
 * squash every bar into the first bin.
 *
 * A ride time of nineteen hours is a data error, but it is a real row, so the
 * end bins still count what falls beyond a clipped domain and bounds drawn
 * from those bins still reach it — only the axis stops early.
 */
function binnedExtent(sorted: readonly number[]): { high: number; low: number } {
  const minimum = sorted[0];
  const maximum = sorted[sorted.length - 1];
  const offset = Math.floor((sorted.length - 1) * CLIP_QUANTILE);
  const low = sorted[offset];
  const high = sorted[sorted.length - 1 - offset];
  const clip =
    high > low && high - low < (maximum - minimum) * CLIP_THRESHOLD;
  return clip ? { high, low } : { high: maximum, low: minimum };
}

/**
 * Bin a column's values for display, or `null` when none of them are numbers.
 *
 * Callers pass values already in filter units, so a percent column bins over
 * 0–100 exactly as its filter box reads.
 */
export function buildHistogram(
  values: Iterable<number | null | undefined>,
): Histogram | null {
  const numbers: number[] = [];
  let missingCount = 0;
  let integral = true;
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      missingCount += 1;
      continue;
    }
    numbers.push(value);
    integral &&= Number.isInteger(value);
  }
  if (numbers.length === 0) {
    return null;
  }

  const sorted = [...numbers].sort((a, b) => a - b);
  const { high, low } = binnedExtent(sorted);
  const span = high - low;
  const step = span > 0 ? niceStep(span, integral) : 1;
  const precision = Math.max(0, -Math.floor(Math.log10(step)));
  const start = roundTo(Math.floor(low / step) * step, precision);
  const unroundedEnd = Math.ceil(high / step) * step;
  const end = roundTo(unroundedEnd > start ? unroundedEnd : start + step, precision);
  const binCount = Math.min(
    MAXIMUM_BIN_COUNT,
    Math.max(1, Math.round((end - start) / step)),
  );

  const bins = Array.from({ length: binCount }, (_unused, index) => ({
    count: 0,
    end: roundTo(start + (index + 1) * step, precision),
    start: roundTo(start + index * step, precision),
  }));
  bins[binCount - 1].end = end;
  for (const value of numbers) {
    const index = Math.min(
      binCount - 1,
      Math.max(0, Math.floor((value - start) / step)),
    );
    bins[index].count += 1;
  }

  return {
    bins,
    end,
    maximum: sorted[sorted.length - 1],
    maxCount: Math.max(...bins.map((bin) => bin.count)),
    minimum: sorted[0],
    missingCount,
    precision,
    start,
    step,
    valueCount: numbers.length,
  };
}

/*
 * What a bar holds, which brushing, highlighting, and the tooltips all read:
 * bins are half-open, `[start, end)`, except the last, which is closed so
 * that the largest value lands somewhere. The end bars also hold the
 * outliers a clipped axis stops short of. An integer column's bar holds
 * whole numbers, so its last value sits one below an exclusive edge.
 */

function isLastBin(histogram: Histogram, index: number): boolean {
  return index === histogram.bins.length - 1;
}

/** Whether the first bar also holds outliers below the axis. */
function clippedBelow(histogram: Histogram): boolean {
  return histogram.start > histogram.minimum;
}

/** Whether the last bar also holds outliers above the axis. */
function clippedAbove(histogram: Histogram): boolean {
  return histogram.end < histogram.maximum;
}

/** The largest value a bar names, as its label and a brush over it state it. */
export function binLast(
  histogram: Histogram,
  index: number,
  integer = false,
): number {
  const bin = histogram.bins[index];
  return integer && !isLastBin(histogram, index) ? bin.end - 1 : bin.end;
}

/**
 * Bounds spanning a run of bars, inclusive at both ends.
 *
 * A run reaching either end of the axis releases that side, so the outliers
 * an end bar holds stay selected with it.
 */
export function boundsFromBins(
  histogram: Histogram,
  first: number,
  second: number,
  integer = false,
): NumericRange {
  const low = Math.min(first, second);
  const high = Math.max(first, second);
  return {
    lower: low === 0 ? Number.NEGATIVE_INFINITY : histogram.bins[low].start,
    upper: isLastBin(histogram, high)
      ? Number.POSITIVE_INFINITY
      : binLast(histogram, high, integer),
  };
}

/**
 * Whether bounds reach a value a bar holds, so that it draws as selected.
 *
 * A continuous bar is lit only when the bounds cross into it, since bounds
 * that merely end on its start share a single value with it. An integer
 * bar's start is a whole number the bounds can name outright, so `3..3`
 * lights the bar spanning 3 to 4.
 */
export function overlapsBin(
  bounds: NumericRange,
  histogram: Histogram,
  index: number,
  integer = false,
): boolean {
  const bin = histogram.bins[index];
  const lowest =
    index === 0 && clippedBelow(histogram) ? Number.NEGATIVE_INFINITY : bin.start;
  const reachesDown = integer ? bounds.upper >= lowest : bounds.upper > lowest;
  const reachesUp = isLastBin(histogram, index)
    ? clippedAbove(histogram) || bounds.lower <= bin.end
    : bounds.lower < bin.end;
  return reachesDown && reachesUp;
}

/** Name the values a bar holds, including the outliers an end bar absorbs. */
export function describeBin(
  histogram: Histogram,
  index: number,
  integer: boolean,
  format: (value: number) => string,
): string {
  const bin = histogram.bins[index];
  const last = binLast(histogram, index, integer);
  if (index === 0 && clippedBelow(histogram)) {
    return `${format(last)} and below`;
  }
  if (isLastBin(histogram, index) && clippedAbove(histogram)) {
    return `${format(bin.start)} and above`;
  }
  return last === bin.start
    ? format(bin.start)
    : `${format(bin.start)} to ${format(last)}`;
}

/** Whether bounds restrict a column at all, or should clear its filter. */
export function restricts(bounds: NumericRange): boolean {
  return Number.isFinite(bounds.lower) || Number.isFinite(bounds.upper);
}

/**
 * Read one end of a range from a filter box, or `null` when it is not a number.
 *
 * An emptied box releases its end rather than restricting it to nothing, and
 * text that is not a number leaves the bound as it was.
 */
export function parseBound(edge: "lower" | "upper", text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") {
    return edge === "lower"
      ? Number.NEGATIVE_INFINITY
      : Number.POSITIVE_INFINITY;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Bounds with one end replaced by a typed value.
 *
 * A bound typed past the other one wins and releases it, so a minimum above
 * the maximum becomes a minimum alone rather than a range holding nothing.
 */
export function withBound(
  bounds: NumericRange,
  edge: "lower" | "upper",
  value: number,
): NumericRange {
  const next = { ...bounds, [edge]: value };
  if (next.lower > next.upper) {
    return edge === "lower"
      ? { lower: value, upper: Number.POSITIVE_INFINITY }
      : { lower: Number.NEGATIVE_INFINITY, upper: value };
  }
  return next;
}

/** Summarise bounds for a collapsed filter control. */
export function describeBounds(
  bounds: NumericRange,
  format: (value: number) => string,
): string {
  const restrictsLower = Number.isFinite(bounds.lower);
  const restrictsUpper = Number.isFinite(bounds.upper);
  if (restrictsLower && restrictsUpper) {
    return `${format(bounds.lower)}–${format(bounds.upper)}`;
  }
  if (restrictsLower) {
    return `≥ ${format(bounds.lower)}`;
  }
  if (restrictsUpper) {
    return `≤ ${format(bounds.upper)}`;
  }
  return "Any";
}
