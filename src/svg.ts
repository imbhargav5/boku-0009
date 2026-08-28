const allowedTags = new Set([
  "svg",
  "g",
  "path",
  "circle",
  "ellipse",
  "rect",
  "polygon",
  "polyline",
  "line",
  "title",
  "desc"
]);

const forbiddenFragments = [
  /<\/?(?:script|style|foreignObject|iframe|object|embed|image|use|animate(?:\w*)?)(?:\s|>)/i,
  /\son\w+\s*=/i,
  /\s(?:href|xlink:href)\s*=/i,
  /url\s*\(/i
];

/**
 * Reject SVG features that can execute code, load data, or make the rendering
 * non-deterministic. This is an allow-list, not a general-purpose SVG parser.
 */
export function validateSvg(input: string): string {
  const svg = input.trim();

  if (!svg.startsWith("<svg") || !svg.endsWith("</svg>")) {
    throw new Error("Artist output must be one complete <svg> document.");
  }

  for (const pattern of forbiddenFragments) {
    if (pattern.test(svg)) {
      throw new Error(`Artist SVG contains a forbidden construct: ${pattern}`);
    }
  }

  for (const tag of svg.matchAll(/<\/?\s*([a-zA-Z][\w:-]*)\b/g)) {
    if (!allowedTags.has(tag[1])) {
      throw new Error(`Artist SVG uses unsupported element <${tag[1]}>.`);
    }
  }

  if (!/\bviewBox\s*=\s*["'][^"']+["']/i.test(svg)) {
    throw new Error("Artist SVG must include a viewBox for deterministic rendering.");
  }

  const viewBox = svg.match(/\bviewBox\s*=\s*["']\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*["']/i);
  if (!viewBox || viewBox.slice(1).some((value) => Number.isNaN(Number(value)))) {
    throw new Error("Artist SVG must use a numeric viewBox.");
  }
  // A text model can occasionally emit runaway path coordinates that render as
  // a solid block or push the subject off-canvas. Reject those candidates so a
  // retry can produce a bounded drawing instead of poisoning the trajectory.
  for (const path of svg.matchAll(/\bd\s*=\s*["']([^"']*)["']/gi)) {
    const numbers = path[1].match(/-?\d+(?:\.\d+)?/g)?.map(Number) ?? [];
    if (numbers.length > 160) throw new Error("Artist SVG path is too complex; use a compact bounded construction.");
    if (numbers.some((number) => number < 0 || number > 512)) {
      throw new Error("Artist SVG path contains coordinates outside the 0..512 canvas.");
    }
  }

  return svg;
}

export function extractSvg(response: string): string {
  const match = response.match(/<svg\b[\s\S]*?<\/svg>/i);
  if (!match) {
    throw new Error("The Artist did not return SVG.");
  }
  return validateSvg(match[0]);
}

export function fixtureSvg(quality: number, variant: number): string {
  const q = Math.max(0, Math.min(1, quality));
  const shift = (variant % 5) * 3 - 6;
  const eyeY = 245 - Math.round(q * 9) + shift / 3;
  const jawWidth = 108 - Math.round(q * 16) + Math.abs(shift);
  const hairPeak = 65 + Math.round((1 - q) * 46);
  const eyeSize = 18 - Math.round(q * 5);
  const mouthCurve = 355 + Math.round((1 - q) * 12);

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="Original spiky-haired avatar">
  <rect width="512" height="512" fill="#07111f"/>
  <circle cx="256" cy="256" r="215" fill="#102a43"/>
  <path id="neck" d="M210 372 L210 424 L302 424 L302 372" fill="#cc8d6f" stroke="#152433" stroke-width="8"/>
  <path id="jacket" d="M114 512 C125 412 186 390 256 390 C326 390 387 412 398 512" fill="#e8782b" stroke="#152433" stroke-width="10"/>
  <path id="face" d="M${256 - jawWidth} 186 C${256 - jawWidth - 6} 281 ${256 - jawWidth + 32} 380 256 398 C${256 + jawWidth - 32} 380 ${256 + jawWidth + 6} 281 ${256 + jawWidth} 186 C350 124 162 124 ${256 - jawWidth} 186 Z" fill="#d99a79" stroke="#152433" stroke-width="10"/>
  <path id="hair" d="M${150 + shift} 209 L${128 + shift} ${hairPeak + 66} L194 112 L206 ${hairPeak} L246 104 L282 ${hairPeak - 16} L310 109 L364 ${hairPeak + 48} L349 207 C321 169 285 155 256 157 C218 153 179 172 ${150 + shift} 209 Z" fill="#f4c542" stroke="#152433" stroke-width="10" stroke-linejoin="round"/>
  <path d="M177 191 L206 128 M225 168 L246 104 M280 167 L282 75 M321 183 L359 113" fill="none" stroke="#b8791d" stroke-width="8" stroke-linecap="round"/>
  <path id="left-brow" d="M171 ${eyeY - 35} Q205 ${eyeY - 57} 231 ${eyeY - 33}" fill="none" stroke="#152433" stroke-width="12" stroke-linecap="round"/>
  <path id="right-brow" d="M281 ${eyeY - 33} Q307 ${eyeY - 57} 341 ${eyeY - 35}" fill="none" stroke="#152433" stroke-width="12" stroke-linecap="round"/>
  <ellipse id="left-eye" cx="205" cy="${eyeY}" rx="${eyeSize + 4}" ry="${eyeSize + 8}" fill="#f7fafc" stroke="#152433" stroke-width="7"/>
  <ellipse id="right-eye" cx="307" cy="${eyeY}" rx="${eyeSize + 4}" ry="${eyeSize + 8}" fill="#f7fafc" stroke="#152433" stroke-width="7"/>
  <circle cx="205" cy="${eyeY + 2}" r="${Math.max(6, eyeSize - 5)}" fill="#152433"/>
  <circle cx="307" cy="${eyeY + 2}" r="${Math.max(6, eyeSize - 5)}" fill="#152433"/>
  <path id="nose" d="M256 252 L244 304 L267 304" fill="none" stroke="#a96d55" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  <path id="mouth" d="M218 ${mouthCurve} Q256 ${mouthCurve + Math.round(q * 7)} 294 ${mouthCurve}" fill="none" stroke="#152433" stroke-width="8" stroke-linecap="round"/>
  <path d="M151 293 L178 304 M361 293 L334 304" fill="none" stroke="#a96d55" stroke-width="8" stroke-linecap="round"/>
</svg>`;
}

/** Compact scaffold shown to the Artist so small coding models start from a readable figure. */
export function artistScaffoldSvg(): string {
  return `<svg viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#ffffff"/>
  <path id="shoulders" d="M92 512Q112 398 256 388Q400 398 420 512Z" fill="#e87522" stroke="#1b2430" stroke-width="10"/>
  <path id="undershirt" d="M205 398L256 468L307 398L286 512H226Z" fill="#24599b"/>
  <path id="neck" d="M214 340V410Q256 432 298 410V340Z" fill="#d79570" stroke="#1b2430" stroke-width="8"/>
  <path id="face" d="M158 178Q164 118 256 108Q348 118 354 178V284Q340 370 256 392Q172 370 158 284Z" fill="#e7a77f" stroke="#1b2430" stroke-width="10"/>
  <path id="hair" d="M144 202L92 132L174 148L136 54L218 112L256 18L276 108L350 46L324 142L420 92L354 210Q322 154 256 154Q190 154 144 202Z" fill="#151922" stroke="#1b2430" stroke-width="10" stroke-linejoin="round"/>
  <path id="left-brow" d="M180 218Q210 198 236 216" fill="none" stroke="#1b2430" stroke-width="12" stroke-linecap="round"/>
  <path id="right-brow" d="M276 216Q302 198 332 218" fill="none" stroke="#1b2430" stroke-width="12" stroke-linecap="round"/>
  <ellipse id="left-eye" cx="210" cy="246" rx="24" ry="16" fill="#ffffff" stroke="#1b2430" stroke-width="7"/><circle cx="210" cy="246" r="8" fill="#1b2430"/>
  <ellipse id="right-eye" cx="302" cy="246" rx="24" ry="16" fill="#ffffff" stroke="#1b2430" stroke-width="7"/><circle cx="302" cy="246" r="8" fill="#1b2430"/>
  <path id="nose" d="M256 252L244 302L268 302" fill="none" stroke="#9b604c" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  <path id="mouth" d="M220 334Q256 350 292 334" fill="none" stroke="#1b2430" stroke-width="8" stroke-linecap="round"/>
</svg>`;
}
