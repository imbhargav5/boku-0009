import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { collectManifest, collectionRunName } from "./collect.js";
import { runLoop } from "./loop.js";
import {
  FixtureArtist,
  FixtureJudge,
  OllamaArtist,
  OllamaJudge,
  OllamaReferenceAnalyzer,
  OllamaVisionArtist,
  OpenAICompatibleArtist,
  OpenAICompatibleJudge,
  OpenAICompatibleReferenceAnalyzer
} from "./providers.js";

type ParsedArgs = Record<string, string | boolean>;

function parseArgs(argv: string[]): ParsedArgs {
  const result: ParsedArgs = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) result[key] = true;
    else {
      result[key] = value;
      index += 1;
    }
  }
  return result;
}

function value(args: ParsedArgs, key: string, fallback?: string): string | undefined {
  const item = args[key];
  return typeof item === "string" ? item : fallback;
}

function runId(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function localConfig(prefix: "ARTIST" | "JUDGE") {
  const endpoint = process.env[`${prefix}_ENDPOINT`];
  const model = process.env[`${prefix}_MODEL`];
  if (!endpoint || !model) {
    throw new Error(`${prefix}_ENDPOINT and ${prefix}_MODEL must be set for a live local-model run. Run \"npm run demo\" to test the pipeline without models.`);
  }
  return { endpoint, model, apiKey: process.env[`${prefix}_API_KEY`] };
}

async function main(): Promise<void> {
  const [command = "help", ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);

  if (command === "help" || command === "--help") {
    console.log("Usage:\n  npm run demo\n  npm run ollama -- --reference <image> --prompt <text> [--artist-vision] [--candidates 4] [--rounds 3] [--out artifacts/run]\n  npm run loop -- --reference <image> --prompt <text> [--candidates 4] [--rounds 3] [--out artifacts/run]\n  npm run collect -- [--manifest data/reference-manifest.jsonl] [--split train] [--artist-vision] [--candidates 4] [--rounds 2] [--max-images N] [--out artifacts/collection-run]");
    return;
  }

  if (command === "collect") {
    const manifestPath = resolve(value(args, "manifest", "data/reference-manifest.jsonl")!);
    if (!existsSync(manifestPath)) throw new Error(`Manifest file does not exist: ${manifestPath}`);
    const outputDirectory = resolve(value(args, "out", join("artifacts", collectionRunName()))!);
    const ollamaHost = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";
    const maxImagesRaw = value(args, "max-images");
    const startRaw = value(args, "start", "0");
    const summary = await collectManifest({
      manifestPath,
      outputDirectory,
      split: value(args, "split", "train"),
      candidatesPerRound: Number(value(args, "candidates", "4")),
      rounds: Number(value(args, "rounds", "2")),
      ...(maxImagesRaw === undefined ? {} : { maxImages: Number(maxImagesRaw) }),
      start: Number(startRaw),
      baseDirectory: resolve(value(args, "base", ".")!),
      artist: args["artist-vision"] === true
        ? new OllamaVisionArtist(ollamaHost, value(args, "vision-model", process.env.VISION_MODEL ?? process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")!)
        : new OllamaArtist(ollamaHost, value(args, "artist-model", process.env.ARTIST_MODEL ?? "qwen2.5-coder:latest")!),
      judge: new OllamaJudge(ollamaHost, value(args, "judge-model", process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")!),
      analyzer: new OllamaReferenceAnalyzer(ollamaHost, value(args, "vision-model", process.env.VISION_MODEL ?? process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")!)
    });
    console.log(`Collection complete: ${summary.completed} completed, ${summary.skipped} skipped, ${summary.failed} failed.`);
    console.log(`Artifacts: ${summary.outputDirectory}`);
    return;
  }

  const fixture = command === "demo" || args.fixture === true;
  const useOllama = command === "ollama";
  const referencePath = fixture
    ? resolve("fixtures/original-avatar-reference.svg")
    : value(args, "reference");
  const prompt = value(args, "prompt", "Draw an original front-facing, neutral-expression spiky-haired avatar with head and shoulders.");

  if (!referencePath) throw new Error("--reference is required for a live run.");
  if (!existsSync(referencePath)) throw new Error(`Reference file does not exist: ${referencePath}`);

  const outputDirectory = resolve(value(args, "out", join("artifacts", runId()))!);
  const ollamaHost = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";
  const artist = fixture
    ? new FixtureArtist()
    : useOllama
      ? args["artist-vision"] === true
        ? new OllamaVisionArtist(ollamaHost, process.env.VISION_MODEL ?? process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")
        : new OllamaArtist(ollamaHost, process.env.ARTIST_MODEL ?? "qwen2.5-coder:latest")
      : new OpenAICompatibleArtist(localConfig("ARTIST"));
  const judge = fixture
    ? new FixtureJudge()
    : useOllama
      ? new OllamaJudge(ollamaHost, process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")
      : new OpenAICompatibleJudge(localConfig("JUDGE"));
  const analyzer = fixture
    ? undefined
    : useOllama
      ? new OllamaReferenceAnalyzer(ollamaHost, process.env.VISION_MODEL ?? process.env.JUDGE_MODEL ?? "qwen2.5vl:3b")
      : new OpenAICompatibleReferenceAnalyzer(localConfig("JUDGE"));
  const result = await runLoop({
    prompt: prompt!,
    referencePath,
    candidatesPerRound: Number(value(args, "candidates", "4")),
    rounds: Number(value(args, "rounds", fixture ? "3" : "3")),
    outputDirectory,
    artist,
    judge,
    analyzer
  });
  console.log(`Completed ${result.outputDirectory}\nWinner: ${result.winner.id} (${result.winner.score.total}/100)\nPNG: ${result.winner.pngPath}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
