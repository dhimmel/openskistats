import { build, context } from "esbuild";

const watch = process.argv.includes("--watch");
const options = {
  bundle: true,
  entryPoints: {
    "common/table.bundle": "tables/src/main.tsx",
    "index/snowflake": "index/snowflake.ts",
  },
  format: "esm",
  logLevel: "info",
  minify: !watch,
  outdir: "../data/webapp",
  sourcemap: true,
};

if (watch) {
  const buildContext = await context(options);
  await buildContext.watch();
} else {
  await build(options);
}
