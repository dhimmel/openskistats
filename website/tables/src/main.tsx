import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { SkiAreaTable } from "./SkiAreaTable";
import {
  readSkiAreaDocument,
  SkiAreaContractError,
  type SkiAreaDocument,
} from "./types";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; document: SkiAreaDocument }
  | { status: "error"; message: string };

function SkiAreaTableLoader({ source }: { source: string }) {
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
        const document = readSkiAreaDocument(await response.json());
        setState({ document, status: "ready" });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        const message =
          error instanceof SkiAreaContractError
            ? error.message
            : `Unable to load ski-area data from ${source}. ${error instanceof Error ? error.message : String(error)}`;
        setState({ message, status: "error" });
      }
    }

    void load();
    return () => controller.abort();
  }, [attempt, source]);

  if (state.status === "loading") {
    return <p className="oss-table-loading">Loading ski-area data…</p>;
  }
  if (state.status === "error") {
    return (
      <div className="oss-table-error" role="alert">
        <strong>The ski-area table could not be loaded.</strong>
        <p>{state.message}</p>
        <button onClick={() => setAttempt((value) => value + 1)} type="button">
          Try again
        </button>
      </div>
    );
  }
  return <SkiAreaTable document={state.document} />;
}

const mount = document.querySelector<HTMLElement>("#ski-area-table");
if (mount) {
  const source = mount.dataset.source;
  createRoot(mount).render(
    <StrictMode>
      {source ? (
        <SkiAreaTableLoader source={source} />
      ) : (
        <div className="oss-table-error" role="alert">
          The ski-area table is missing its data source URL.
        </div>
      )}
    </StrictMode>,
  );
}
