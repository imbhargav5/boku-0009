import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import type {
  ArtistProvider,
  DrawingRequest,
  GeneratedDrawing,
  JudgeProvider,
  JudgeRequest,
  ReferenceAnalysisRequest,
  ReferenceAnalyzer,
  ScoreReport
} from "./contracts.js";
import { referenceAnalysisSchema, scoreReportSchema } from "./contracts.js";
import { artistScaffoldSvg, extractSvg, fixtureSvg } from "./svg.js";

type LocalModelConfig = {
  endpoint: string;
  model: string;
  apiKey?: string;
};

type ChatResponse = {
  choices?: Array<{ message?: { content?: string | Array<{ text?: string }> } }>;
};

type OllamaResponse = {
  message?: { content?: string };
};

const modelRequestTimeoutMs = Number(process.env.MODEL_REQUEST_TIMEOUT_MS ?? "120000");

function requestSignal(): AbortSignal {
  const timeout = Number.isFinite(modelRequestTimeoutMs) && modelRequestTimeoutMs > 0 ? modelRequestTimeoutMs : 120000;
  return AbortSignal.timeout(timeout);
}

async function localChat(config: LocalModelConfig, messages: unknown[], temperature = 0.65): Promise<string> {
  const endpoint = `${config.endpoint.replace(/\/$/, "")}/chat/completions`;
  const response = await fetch(endpoint, {
    method: "POST",
    signal: requestSignal(),
    headers: {
      "content-type": "application/json",
      ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {})
    },
    body: JSON.stringify({ model: config.model, temperature, messages })
  });

  if (!response.ok) {
    throw new Error(`Local model request to ${endpoint} failed: ${response.status} ${await response.text()}`);
  }

  const payload = (await response.json()) as ChatResponse;
  const content = payload.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part) => part.text ?? "").join("\n");
  throw new Error("Local model response did not contain message content.");
}

function dataUrl(buffer: Buffer): string {
  return `data:image/png;base64,${buffer.toString("base64")}`;
}

function parseJsonObject(raw: string, role = "Model"): unknown {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1] ?? raw;
  const object = fenced.match(/\{[\s\S]*\}/)?.[0];
  if (!object) throw new Error(`${role} did not return a JSON object.`);
  return JSON.parse(object);
}

async function ollamaChat(host: string, model: string, messages: unknown[], temperature = 0.65): Promise<string> {
  const endpoint = `${host.replace(/\/$/, "")}/api/chat`;
  const response = await fetch(endpoint, {
    method: "POST",
    signal: requestSignal(),
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, stream: false, options: { temperature }, messages })
  });

  if (!response.ok) {
    throw new Error(`Ollama request to ${endpoint} failed: ${response.status} ${await response.text()}`);
  }

  const payload = (await response.json()) as OllamaResponse;
  if (!payload.message?.content) throw new Error("Ollama response did not contain message content.");
  return payload.message.content;
}

function visualBrief(request: DrawingRequest): string {
  if (!request.referenceAnalysis) {
    return "No image-derived brief is available. Prioritize a clear, centered silhouette and the named visual attributes in the task.";
  }
  return `Image-derived reference brief (follow this closely; do not invent unrelated details):\n${JSON.stringify(request.referenceAnalysis, null, 2)}`;
}

function scaffoldBrief(): string {
  return `Safe SVG scaffold (adapt this instead of starting from a blank canvas):\n${artistScaffoldSvg()}`;
}

const artistInstructions = "You are the Artist in an iterative drawing system. Return only one self-contained SVG, no Markdown and no explanation. Use viewBox='0 0 512 512'. All geometry coordinates must stay between 0 and 512; never emit negative or four-digit coordinates. Keep the SVG compact (under 2,500 characters, no more than 14 visible shapes) and use simple closed paths, polygons, circles, and ellipses rather than repeated coordinate sequences. Use a centered, readable subject that fills the canvas. Allowed tags: svg, g, path, circle, ellipse, rect, polygon, polyline, line, title, desc. Never use style, script, image, use, filters, URLs, event handlers, text, CSS, or external assets. Use actual colored fills and simple facial construction: hair, face, neck, clothing, eyebrows, eyes, nose, and mouth. Give important shapes clear id attributes such as hair, face, left-eye, right-eye, jaw. The mustMatch cues in the visual brief take priority over any contradictory avoid cue.";

export class OpenAICompatibleArtist implements ArtistProvider {
  constructor(private readonly config: LocalModelConfig) {}

  async generate(request: DrawingRequest): Promise<GeneratedDrawing> {
    const prior = request.previousSvg
      ? `Previous selected SVG:\n${request.previousSvg}\n\nJudge feedback:\n${request.feedback?.critique.map((item) => `- ${item}`).join("\n")}\nNext step: ${request.feedback?.nextStep}`
      : "There is no previous drawing. Create a strong first candidate.";

    const output = await localChat(this.config, [
      {
        role: "system",
        content: artistInstructions
      },
      {
        role: "user",
        content: `Task: ${request.prompt}\n${visualBrief(request)}\n${request.previousSvg ? "" : `${scaffoldBrief()}\n`}Round ${request.round}, candidate ${request.attempt}.\n${prior}\nProduce a materially distinct candidate that addresses the critique.`
      }
    ]);

    return { svg: extractSvg(output), model: this.config.model };
  }
}

export class OpenAICompatibleJudge implements JudgeProvider {
  constructor(private readonly config: LocalModelConfig) {}

  async score(request: JudgeRequest): Promise<ScoreReport> {
    const [reference, candidate] = await Promise.all([readFile(request.referencePng), readFile(request.candidatePng)]);
    const output = await localChat(this.config, [
      {
        role: "system",
        content: "You are the Teacher in a drawing-improvement system. Compare the first image (reference) with the second (candidate) for the requested original subject. Return only valid JSON with this exact shape: { total: 0-100 number, dimensions: { silhouette: 0-100, facialFeatures: 0-100, proportion: 0-100, composition: 0-100, style: 0-100 }, critique: [1 to 5 precise, actionable strings], nextStep: one concise instruction }. Be strict, consistent, and do not identify copyrighted characters."
      },
      {
        role: "user",
        content: [
          { type: "text", text: `Prompt: ${request.prompt}\n${request.referenceAnalysis ? `Image-derived brief:\n${JSON.stringify(request.referenceAnalysis, null, 2)}\n` : ""}Reference file: ${basename(request.referencePng)}\nCandidate SVG:\n${request.candidateSvg}` },
          { type: "image_url", image_url: { url: dataUrl(reference) } },
          { type: "image_url", image_url: { url: dataUrl(candidate) } }
        ]
      }
    ]);

    return scoreReportSchema.parse(parseJsonObject(output));
  }
}

const referenceAnalysisInstructions = "You are a visual analyst preparing a precise brief for a text-only SVG artist. Do not name or identify any copyrighted character. Describe only visible, actionable attributes. Return only valid JSON with this exact shape: { subject: string, framing: string, pose: string, silhouette: string, hair: string, face: string, clothing: string, palette: string[], lineAndRendering: string, background: string, mustMatch: string[], avoid: string[] }. The mustMatch list should contain the most important identity and composition cues; avoid should list common mistakes.";

export class OpenAICompatibleReferenceAnalyzer implements ReferenceAnalyzer {
  constructor(private readonly config: LocalModelConfig) {}

  async analyze(request: ReferenceAnalysisRequest) {
    const reference = await readFile(request.referencePng);
    const output = await localChat(this.config, [
      { role: "system", content: referenceAnalysisInstructions },
      {
        role: "user",
        content: [
          { type: "text", text: `Task: ${request.prompt}\nAnalyze the attached reference for an SVG artist.` },
          { type: "image_url", image_url: { url: dataUrl(reference) } }
        ]
      }
    ], 0.15);
    return referenceAnalysisSchema.parse(parseJsonObject(output, "Reference analyzer"));
  }
}

/** Uses Ollama's native image API instead of its OpenAI compatibility layer. */
export class OllamaArtist implements ArtistProvider {
  constructor(private readonly host: string, private readonly model: string) {}

  async generate(request: DrawingRequest): Promise<GeneratedDrawing> {
    const prior = request.previousSvg
      ? `Previous selected SVG:\n${request.previousSvg}\n\nJudge feedback:\n${request.feedback?.critique.map((item) => `- ${item}`).join("\n")}\nNext step: ${request.feedback?.nextStep}`
      : "There is no previous drawing. Create a strong first candidate.";
    const output = await ollamaChat(this.host, this.model, [
      {
        role: "system",
        content: artistInstructions
      },
      {
        role: "user",
        content: `Task: ${request.prompt}\n${visualBrief(request)}\n${request.previousSvg ? "" : `${scaffoldBrief()}\n`}Round ${request.round}, candidate ${request.attempt}.\n${prior}\nProduce a materially distinct candidate that addresses the critique.`
      }
    ]);
    return { svg: extractSvg(output), model: this.model };
  }
}

/** Uses the vision model as an image-conditioned Artist when visual fidelity matters more than code specialization. */
export class OllamaVisionArtist implements ArtistProvider {
  constructor(private readonly host: string, private readonly model: string) {}

  async generate(request: DrawingRequest): Promise<GeneratedDrawing> {
    if (!request.referencePng) throw new Error("The vision Artist requires a normalized reference PNG.");
    const reference = await readFile(request.referencePng);
    const prior = request.previousSvg
      ? `Previous selected SVG:\n${request.previousSvg}\n\nJudge feedback:\n${request.feedback?.critique.map((item) => `- ${item}`).join("\n")}\nNext step: ${request.feedback?.nextStep}`
      : "There is no previous drawing. Create a strong first candidate.";
    const output = await ollamaChat(this.host, this.model, [
      { role: "system", content: artistInstructions },
      {
        role: "user",
        content: `Task: ${request.prompt}\n${visualBrief(request)}\n${request.previousSvg ? "" : `${scaffoldBrief()}\n`}Round ${request.round}, candidate ${request.attempt}.\n${prior}\nUse the attached reference image as the primary visual source. Reconstruct its pose, hair silhouette, face, clothing, palette, and framing as a clean original SVG. Produce a materially distinct candidate that addresses the critique.`,
        images: [reference.toString("base64")]
      }
    ], 0.45);
    return { svg: extractSvg(output), model: this.model };
  }
}

export class OllamaReferenceAnalyzer implements ReferenceAnalyzer {
  constructor(private readonly host: string, private readonly model: string) {}

  async analyze(request: ReferenceAnalysisRequest) {
    const reference = await readFile(request.referencePng);
    const output = await ollamaChat(this.host, this.model, [
      { role: "system", content: referenceAnalysisInstructions },
      {
        role: "user",
        content: `Task: ${request.prompt}\nAnalyze the first image as a visual reference for a text-only SVG artist. Do not name the subject; describe its visible construction precisely.`,
        images: [reference.toString("base64")]
      }
    ], 0.15);
    return referenceAnalysisSchema.parse(parseJsonObject(output, "Reference analyzer"));
  }
}

export class OllamaJudge implements JudgeProvider {
  constructor(private readonly host: string, private readonly model: string) {}

  async score(request: JudgeRequest): Promise<ScoreReport> {
    const [reference, candidate] = await Promise.all([readFile(request.referencePng), readFile(request.candidatePng)]);
    const output = await ollamaChat(this.host, this.model, [
      {
        role: "system",
        content: "You are the Teacher in a drawing-improvement system. Compare the first image (reference) with the second (candidate) for the requested original subject. Return only valid JSON with this exact shape: { total: 0-100 number, dimensions: { silhouette: 0-100, facialFeatures: 0-100, proportion: 0-100, composition: 0-100, style: 0-100 }, critique: [1 to 5 precise, actionable strings], nextStep: one concise instruction }. Be strict, consistent, and do not identify copyrighted characters."
      },
      {
        role: "user",
        content: `Prompt: ${request.prompt}\n${request.referenceAnalysis ? `Image-derived brief:\n${JSON.stringify(request.referenceAnalysis, null, 2)}\n` : ""}The first image is the reference and the second image is the candidate.\nCandidate SVG:\n${request.candidateSvg}`,
        images: [reference.toString("base64"), candidate.toString("base64")]
      }
    ]);
    return scoreReportSchema.parse(parseJsonObject(output));
  }
}

/** A deterministic, offline provider used to verify rendering and orchestration. */
export class FixtureArtist implements ArtistProvider {
  async generate(request: DrawingRequest): Promise<GeneratedDrawing> {
    const quality = Math.min(0.95, 0.35 + request.round * 0.12 + (request.attempt % 4) * 0.08);
    return { svg: fixtureSvg(quality, request.attempt), model: "offline-fixture-artist" };
  }
}

export class FixtureJudge implements JudgeProvider {
  async score(request: JudgeRequest): Promise<ScoreReport> {
    const index = Number(request.candidatePng.match(/candidate-(\d+)/)?.[1] ?? 1);
    const round = Number(request.candidatePng.match(/round-(\d+)/)?.[1] ?? 1);
    const total = Math.min(96, 46 + round * 11 + (index % 4) * 7);
    return {
      total,
      dimensions: {
        silhouette: Math.min(100, total + 2),
        facialFeatures: Math.max(0, total - 5),
        proportion: total,
        composition: Math.max(0, total - 2),
        style: Math.min(100, total + 4)
      },
      critique: [
        "Make the hair silhouette more decisive and balanced.",
        "Tighten eye size and vertical alignment.",
        "Narrow the lower face slightly while retaining the neck connection."
      ],
      nextStep: "Refine the selected SVG's hair, eyes, and jaw without adding visual noise."
    };
  }
}
