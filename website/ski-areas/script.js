function filterCountry(rows, columnId, filterValue) {
  const filterValueLower = filterValue.toLowerCase();
  return rows.filter(function(row) {
      return (
          (row.values["country"] && row.values["country"].toLowerCase().includes(filterValueLower)) ||
          (row.values["country_code"] && row.values["country_code"].toLowerCase() === filterValueLower) ||
          (row.values["country_emoji"] && row.values["country_emoji"] === filterValue)
      );
  });
}


function filterLatitude(rows, columnId, filterValue) {
  return rows.filter(function(row) {
      const latitude = row.values["latitude"];
      const hemisphere = latitude > 0 ? "north" : "south";

      // Check if filterValue is entirely alphabetic
      const isAlphabetic = /^[a-zA-Z]+$/.test(filterValue);

      // Use matchesNumericFilter for numeric filter values
      if (!isAlphabetic) {
          return matchesNumericFilter(latitude, filterValue);
      }

      // Handle string filter values for "north" or "south"
      if (typeof filterValue === "string") {
          return hemisphere.includes(filterValue.toLowerCase());
      }

      // Default: include all rows if filterValue is invalid
      return true;
  });
}

function formatNumber(value) {
  // Format the number with commas as thousand separators
  return value.toLocaleString(undefined, {
      maximumFractionDigits: 0 // No decimal digits
  });
}

function formatMeters(value) {
  const NARROW_SPACE = '\u202F';
  const formattedNumber = formatNumber(value);
  return `${formattedNumber}${NARROW_SPACE}m`;
}

function sumColumn(column, state) {
  let total = 0;
  state.sortedData.forEach(function(row) {
    total += row[column.id]
  });
  return total;
}

function footerSum(column, state) {
  const total = sumColumn(column, state);
  return `Sum: ${formatNumber(total)}`;
}

function footerSumMeters(column, state) {
  const total = sumColumn(column, state);
  return `Sum: ${formatNumber(total)}`;
}

function footerMinMeters(column, state) {
  const min = Math.min(...state.sortedData.map(row => row[column.id]));
  return `Min: ${formatNumber(min)}`;
}

function footerMaxMeters(column, state) {
  const max = Math.max(...state.sortedData.map(row => row[column.id]));
  return `Max: ${formatNumber(max)}`;
}

function footerDistinctCount(column, state) {
  const distinctValues = new Set(
    state.sortedData
      .map(row => row[column.id])
      .filter(value => value !== null && value !== undefined)
  );
  return `Distinct: ${formatNumber(distinctValues.size)}`;
}

function meanWeighted(column, state) {
  let numerator = 0;
  let denominator = 0;
  state.sortedData.forEach(function(row) {
    numerator += row["combined_vertical"] * row[column.id]
    denominator += row["combined_vertical"]
  });
  return numerator / denominator;
}

function footerMeanWeightedPercent(column, state) {
  return `Wtd. Mean: ${formatNumber(100 * meanWeighted(column, state))}%`;
}



function matchesNumericFilter(value, filterValue) {
    filterValue = filterValue.trim();
    let match;

    // Handle "-" and "-0" as shorthand for <= 0
    if (filterValue === "-" || filterValue === "-0") {
        return value <= 0;
    }

    // Handle basic numbers (>= for positive, <= for negative)
    if ((match = filterValue.match(/^(-?\d+)$/))) {
        const threshold = parseFloat(match[1]);
        return threshold >= 0 ? value >= threshold : value <= threshold;
    }

    // Handle explicit ranges with brackets (inclusive) or parentheses (exclusive)
    if ((match = filterValue.match(/^([\[\(])\s*(-?\d+)?\s*,\s*(-?\d+)?\s*([\]\)])$/))) {
        const lower = match[2] !== undefined ? parseFloat(match[2]) : -Infinity;
        const upper = match[3] !== undefined ? parseFloat(match[3]) : Infinity;
        const lowerInclusive = match[1] === '['; // True if inclusive lower
        const upperInclusive = match[4] === ']'; // True if inclusive upper

        return (
            (lowerInclusive ? value >= lower : value > lower) &&
            (upperInclusive ? value <= upper : value < upper)
        );
    }

    // Default: invalid filterValue returns true
    return true;
}

function filterNumeric(rows, columnId, filterValue) {
  return rows.filter(row => matchesNumericFilter(row.values[columnId], filterValue));
}

function filterPercent(rows, columnId, filterValue) {
  return rows.filter(row => matchesNumericFilter(row.values[columnId] * 100, filterValue));
}


function cellAzimuth(cellInfo) {
  const azimuth = cellInfo.value; // Original azimuth for arrow rotation
  const displayedAzimuth = Math.round(azimuth); // Rounded azimuth for display only

  return `
  <div class="azimuth-arrow-cell" style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
      <svg width="36" height="36" viewBox="0 0 36 36" style="transform: rotate(${azimuth}deg); margin-bottom: 6px;">
          <circle cx="18" cy="18" r="3" fill="black" />
          <line x1="18" y1="18" x2="18" y2="6" stroke="black" stroke-width="3" />
          <polygon points="12,9 18,1.5 24,9" fill="black" />
      </svg>
      <span style="font-size: 12px; color: #333;">${displayedAzimuth}°</span>
  </div>
  `;
}


function cellRose(cellInfo) {
  return `
  <div class="tooltip-container"
       onmouseover="showTooltip(event)"
       onmouseout="hideTooltip(event)">
      <a href="roses-full/${cellInfo.value}.svg" target="_blank">
          <img src="/ski-areas/roses-preview/${cellInfo.value}.svg" alt="Preview Rose" class="hover-preview">
      </a>
      <div class="tooltip-image">
          <img src="/ski-areas/roses-full/${cellInfo.value}.svg" alt="Full Rose">
      </div>
  </div>
  `;
}


function showTooltip(event) {
    const tooltip = event.currentTarget.querySelector('.tooltip-image');
    const rect = event.currentTarget.getBoundingClientRect();
    const tooltipWidth = 300; // Assume tooltip image has a fixed width
    const tooltipHeight = 300; // Assume tooltip image has a fixed height
    const previewHeight = rect.height; // Height of the preview image

    // Position to the left of the preview image
    let left = rect.left - tooltipWidth - 10; // Add a small gap of 10px

    // Center vertically on the preview image
    let top = rect.top + (previewHeight / 2) - (tooltipHeight / 2);

    // Check if the tooltip overflows the left edge of the viewport
    if (left < 0) {
        left = 10; // Adjust to prevent overflow with padding
    }

    // Check if the tooltip overflows the top edge of the viewport
    if (top < 0) {
        top = 10; // Adjust to prevent overflow with padding
    }

    // Check if the tooltip overflows the bottom edge of the viewport
    if (top + tooltipHeight > window.innerHeight) {
        top = window.innerHeight - tooltipHeight - 10; // Adjust to prevent overflow
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.visibility = 'visible';
    tooltip.style.opacity = '1';
}


function hideTooltip(event) {
    const tooltip = event.currentTarget.querySelector('.tooltip-image');
    tooltip.style.visibility = 'hidden';
    tooltip.style.opacity = '0';
}

// Generates HTML for a donut chart given a percentage value (out of 1) and color
// Modified from <https://glin.github.io/reactable/articles/popular-movies/popular-movies.html>
function donutChart(value, color = '#a100bf') {
    // Format as percentage
    const pct = (value * 100).toFixed(0)
    // All units are in rem for relative scaling
    const radius = 1.5
    const diameter = 3.75
    const center = diameter / 2
    const width = 0.25
    const sliceLength = 2 * Math.PI * radius
    const sliceOffset = sliceLength * (1 - value)
    const donutChart = `
      <svg width="${diameter}rem" height="${diameter}rem" style="transform: rotate(-90deg)" focusable="false">
        <circle cx="${center}rem" cy="${center}rem" r="${radius}rem" fill="none" stroke-width="${width}rem" stroke="rgba(0,0,0,0.1)"></circle>
        <circle cx="${center}rem" cy="${center}rem" r="${radius}rem" fill="none" stroke-width="${width}rem" stroke="${color}"
         stroke-dasharray="${sliceLength}rem" stroke-dashoffset="${sliceOffset}rem"></circle>
      </svg>
    `
    const label = `
      <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%)">
        ${pct}%
      </div>
    `
    return `
      <div style="display: inline-flex; position: relative">
        ${donutChart}
        ${label}
      </div>
    `
}
  