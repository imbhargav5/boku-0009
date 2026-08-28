import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import type { ArtistProvider, Candidate, JudgeProvider, ReferenceAnalysis, ReferenceAnalyzer } from "./contracts.js";
import { renderReference, renderSvg } from "./render.js";
import { validateSvg } from "./svg.js";

export type LoopOptions = {
  prompt: string;
  referencePath: string;
  candidatesPerRound: number;
  rounds: number;
  outputDirectory: string;
  artist: ArtistProvider;
  judge: JudgeProvider;
  analyzer?: ReferenceAnalyzer;
};

export type LoopResult = {
  outputDirectory: string;
  winner: Candidate;
};

function candidateId(round: number, attempt: number): string {
  return `round-${String(round).padStart(2, "0")}-candidate-${String(attempt).padStart(2, "0")}`;
}

async function appendTrajectory(path: string, candidate: Candidate): Promise<void> {
  const record = {
    id: candidate.id,
    parentId: candidate.parentId ?? null,
    round: candidate.round,
    attempt: candidate.attempt,
    svg: candidate.svg,
    png: candidate.pngPath,
    score: candidate.score
  };
  await appendFile(path, `${JSON.stringify(record)}\n`, "utf8");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function withRetry<T>(operation: () => Promise<T>, attempts = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function writeAttemptError(path: string, details: Record<string, unknown>): Promise<void> {
  await writeFile(`${path}.error.json`, `${JSON.stringify({
    ...details,
    createdAt: new Date().toISOString()
  }, null, 2)}\n`, "utf8");
}

export async function runLoop(options: LoopOptions): Promise<LoopResult> {
  if (!options.prompt.trim()) throw new Error("A drawing prompt is required.");
  if (!Number.isInteger(options.candidatesPerRound) || options.candidatesPerRound < 2 || options.candidatesPerRound > 8) {
    throw new Error("--candidates must be an integer from 2 through 8.");
  }
  if (!Number.isInteger(options.rounds) || options.rounds < 1 || options.rounds > 20) {
    throw new Error("--rounds must be an integer from 1 through 20.");
  }

  const outputDirectory = resolve(options.outputDirectory);
  const trajectoryPath = join(outputDirectory, "trajectory.jsonl");
  await mkdir(outputDirectory, { recursive: true });
  await writeFile(join(outputDirectory, "config.json"), JSON.stringify({
    prompt: options.prompt,
    reference: resolve(options.referencePath),
    candidatesPerRound: options.candidatesPerRound,
    rounds: options.rounds,
    referenceAnalysis: Boolean(options.analyzer),
    createdAt: new Date().toISOString()
  }, null, 2));

  // Render the reference once so SVG and raster references take the same path.
  const referencePng = join(outputDirectory, "reference.png");
  await renderReference(options.referencePath, referencePng);

  let referenceAnalysis: ReferenceAnalysis | undefined;
  if (options.analyzer) {
    try {
      referenceAnalysis = await withRetry(() => options.analyzer!.analyze({
        prompt: options.prompt,
        referencePng
      }));
      await writeFile(join(outputDirectory, "reference-analysis.json"), `${JSON.stringify(referenceAnalysis, null, 2)}\n`, "utf8");
    } catch (error) {
      await writeAttemptError(join(outputDirectory, "reference-analysis.json"), {
        phase: "reference-analysis",
        error: errorMessage(error)
      });
      throw new Error(`Reference analysis failed: ${errorMessage(error)}`);
    }
  }

  let previous: Candidate | undefined;
  for (let round = 1; round <= options.rounds; round += 1) {
    const roundDirectory = join(outputDirectory, `round-${String(round).padStart(2, "0")}`);
    await mkdir(roundDirectory, { recursive: true });
    const drafts: Array<{ id: string; attempt: number; svg: string; pngPath: string }> = [];
    const candidates: Candidate[] = [];

    // Finish the Artist batch before asking the Judge to load. This minimizes
    // model churn on memory-constrained local machines.
    for (let attempt = 1; attempt <= options.candidatesPerRound; attempt += 1) {
      const id = candidateId(round, attempt);
      const svgPath = join(roundDirectory, `candidate-${String(attempt).padStart(2, "0")}.svg`);
      const pngPath = join(roundDirectory, `candidate-${String(attempt).padStart(2, "0")}.png`);
      try {
        const svg = await withRetry(async () => {
          const generated = await options.artist.generate({
            prompt: options.prompt,
            attempt,
            round,
            referencePng,
            previousSvg: previous?.svg,
            feedback: previous?.score,
            referenceAnalysis
          });
          const validated = validateSvg(generated.svg);
          await writeFile(svgPath, validated, "utf8");
          await renderSvg(validated, pngPath);
          return validated;
        });
        drafts.push({ id, attempt, svg, pngPath });
      } catch (error) {
        await writeAttemptError(svgPath, {
          id,
          round,
          attempt,
          phase: "artist",
          error: errorMessage(error)
        });
      }
    }

    if (drafts.length === 0) {
      throw new Error(`Round ${round} produced no valid Artist candidates.`);
    }

    for (const draft of drafts) {
      try {
        const score = await withRetry(() => options.judge.score({
          prompt: options.prompt,
          referencePng,
          candidatePng: draft.pngPath,
          candidateSvg: draft.svg,
          referenceAnalysis
        }));
        await writeFile(`${draft.pngPath}.score.json`, `${JSON.stringify(score, null, 2)}\n`);
        const candidate = {
          id: draft.id,
          attempt: draft.attempt,
          round,
          svg: draft.svg,
          pngPath: draft.pngPath,
          score,
          parentId: previous?.id
        };
        candidates.push(candidate);
        await appendTrajectory(trajectoryPath, candidate);
      } catch (error) {
        await writeAttemptError(draft.pngPath, {
          id: draft.id,
          round,
          attempt: draft.attempt,
          phase: "judge",
          error: errorMessage(error)
        });
      }
    }

    if (candidates.length === 0) {
      throw new Error(`Round ${round} produced no valid Judge scores.`);
    }

    candidates.sort((left, right) => right.score.total - left.score.total);
    previous = candidates[0];
    await writeFile(join(roundDirectory, "winner.json"), `${JSON.stringify({
      id: previous.id,
      score: previous.score.total,
      png: basename(previous.pngPath),
      parentId: previous.parentId ?? null
    }, null, 2)}\n`);
  }

  if (!previous) throw new Error("Loop completed without a candidate.");
  await writeFile(join(outputDirectory, "winner.json"), `${JSON.stringify({
    id: previous.id,
    score: previous.score.total,
    png: previous.pngPath,
    source: dirname(previous.pngPath)
  }, null, 2)}\n`);
  return { outputDirectory, winner: previous };
}
