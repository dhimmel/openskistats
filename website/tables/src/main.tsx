import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { LiftTable } from "./LiftTable";
import { SkiAreaTable } from "./SkiAreaTable";
import {
  readLiftDocument,
  readSkiAreaDocument,
  TableContractError,
  type LiftDocument,
  type SkiAreaDocument,
} from "./types";

type TableKind = "ski-areas" | "lifts";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; document: unknown }
  | { status: "error"; message: string };

const READERS: Record<TableKind, (value: unknown) => unknown> = {
  "ski-areas": readSkiAreaDocument,
  lifts: readLiftDocument,
};

const NOUNS: Record<TableKind, string> = {
  "ski-areas": "ski-area",
  lifts: "lift",
};

function TableLoader({ kind, source }: { kind: TableKind; source: string }) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });

    async function load() {
      try {
        const response = await fetch(source, {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`The server returned ${response.status} ${response.statusText}.`);
        }
        const document = READERS[kind](await response.json());
        setState({ document, status: "ready" });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        const message =
          error instanceof TableContractError
            ? error.message
            : `Unable to load ${NOUNS[kind]} data from ${source}. ${error instanceof Error ? error.message : String(error)}`;
        setState({ message, status: "error" });
      }
    }

    void load();
    return () => controller.abort();
  }, [attempt, kind, source]);

  if (state.status === "loading") {
    return <p className="oss-table-loading">Loading {NOUNS[kind]} data…</p>;
  }
  if (state.status === "error") {
    return (
      <div className="oss-table-error" role="alert">
        <strong>The {NOUNS[kind]} table could not be loaded.</strong>
        <p>{state.message}</p>
        <button onClick={() => setAttempt((value) => value + 1)} type="button">
          Try again
        </button>
      </div>
    );
  }
  return kind === "lifts" ? (
    <LiftTable document={state.document as LiftDocument} />
  ) : (
    <SkiAreaTable document={state.document as SkiAreaDocument} />
  );
}

for (const mount of document.querySelectorAll<HTMLElement>("[data-oss-table]")) {
  const source = mount.dataset.source;
  const kind: TableKind = mount.dataset.ossTable === "lifts" ? "lifts" : "ski-areas";
  createRoot(mount).render(
    <StrictMode>
      {source ? (
        <TableLoader kind={kind} source={source} />
      ) : (
        <div className="oss-table-error" role="alert">
          The table is missing its data source URL.
        </div>
      )}
    </StrictMode>,
  );
}
