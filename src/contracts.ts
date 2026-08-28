import { z } from "zod";

export const scoreReportSchema = z.object({
  total: z.number().min(0).max(100),
  dimensions: z.object({
    silhouette: z.number().min(0).max(100),
    facialFeatures: z.number().min(0).max(100),
    proportion: z.number().min(0).max(100),
    composition: z.number().min(0).max(100),
    style: z.number().min(0).max(100)
  }),
  critique: z.array(z.string().min(1)).min(1).max(8),
  nextStep: z.string().min(1)
});

export type ScoreReport = z.infer<typeof scoreReportSchema>;

/** A compact, image-derived brief that a text-only SVG Artist can follow. */
export const referenceAnalysisSchema = z.object({
  subject: z.string().min(1),
  framing: z.string().min(1),
  pose: z.string().min(1),
  silhouette: z.string().min(1),
  hair: z.string().min(1),
  face: z.string().min(1),
  clothing: z.string().min(1),
  palette: z.array(z.string().min(1)).min(1).max(12),
  lineAndRendering: z.string().min(1),
  background: z.string().min(1),
  mustMatch: z.array(z.string().min(1)).min(1).max(12),
  avoid: z.array(z.string().min(1)).max(12)
});

export type ReferenceAnalysis = z.infer<typeof referenceAnalysisSchema>;

export type DrawingRequest = {
  prompt: string;
  attempt: number;
  round: number;
  referencePng?: string;
  previousSvg?: string;
  feedback?: ScoreReport;
  referenceAnalysis?: ReferenceAnalysis;
};

export type GeneratedDrawing = {
  svg: string;
  model: string;
};

export type JudgeRequest = {
  prompt: string;
  referencePng: string;
  candidatePng: string;
  candidateSvg: string;
  referenceAnalysis?: ReferenceAnalysis;
};

export interface ArtistProvider {
  generate(request: DrawingRequest): Promise<GeneratedDrawing>;
}

export interface JudgeProvider {
  score(request: JudgeRequest): Promise<ScoreReport>;
}

export type ReferenceAnalysisRequest = {
  prompt: string;
  referencePng: string;
};

export interface ReferenceAnalyzer {
  analyze(request: ReferenceAnalysisRequest): Promise<ReferenceAnalysis>;
}

export type Candidate = {
  id: string;
  attempt: number;
  round: number;
  svg: string;
  pngPath: string;
  score: ScoreReport;
  parentId?: string;
};
