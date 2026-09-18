import { useEffect, useRef } from "react";

/**
 * AI-themed line icons that drift, and scatter away from the cursor.
 *
 * Particles live only in the side gutters outside the message column (or
 * everywhere when the chat is empty). Each frame: apply cursor repulsion,
 * integrate velocity + spin, damp both with friction, wrap at the edges.
 */

/** Target area per particle, in CSS px². */
const DENSITY = 5000;
const MIN_SIZE = 8;
const MAX_SIZE = 18;
/** Extra clearance around the chat zone so icons don't kiss its edge. */
const KEEP_OUT_PAD = 24;

const REPEL_RADIUS = 120;
const REPEL_FORCE = 1.6;

/**
 * Reynolds steering (see natureofcode.com/autonomous-agents):
 * steering = desired - velocity, with the result capped at MAX_FORCE.
 * MAX_SPEED caps how fast a particle travels, MAX_FORCE how sharply it can
 * turn — together they produce smooth, deliberate motion rather than the
 * twitchy drift a per-frame random force gives.
 */
const MAX_SPEED = 0.45;
const MAX_FORCE = 0.035;
/** Cursor shoves may briefly exceed MAX_SPEED; this is the hard ceiling. */
const MAX_BURST = 6;

/**
 * Wander: a circle projected WANDER_DIST ahead of the particle, with a
 * target riding its perimeter. The target's angle changes by at most
 * WANDER_JITTER per frame, so headings stay coherent between frames.
 */
const WANDER_DIST = 28;
const WANDER_RADIUS = 18;
const WANDER_JITTER = 0.32;

/** Distance from a boundary at which the inward steering kicks in. */
const EDGE_MARGIN = 80;
/** Velocity damping, applied only to cursor-imparted burst velocity. */
const FRICTION = 0.96;

const LABELS = ["AI", "LLM", "GPU", "API", "01", "{ }"];
const KINDS = 6 + LABELS.length;

type Props = {
  /**
   * The chat column. Its live bounding box is the zone particles never
   * enter, so the reservation follows the real layout at any width.
   */
  keepOutRef: React.RefObject<HTMLElement | null>;
  className?: string;
};

type Particle = {
  x: number; y: number;
  vx: number; vy: number;
  rot: number; spin: number;
  size: number; kind: number;
  /** Angle of the wander target on its projected circle. */
  wander: number;
  /** Gutter this particle lives in: -1 left of the chat zone, +1 right. */
  side: -1 | 1;
};

export default function AiBackdrop({ keepOutRef, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dark = window.matchMedia("(prefers-color-scheme: dark)");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let particles: Particle[] = [];
    let w = 0, h = 0;
    // Cursor in canvas coords; null when it's outside the canvas.
    let cursor: { x: number; y: number } | null = null;
    let rafId = 0;

    // Chat zone in canvas coords, refreshed on resize. Horizontal only:
    // the column spans the full height, so only x matters.
    let zoneL = 0, zoneR = 0;
    const measureZone = () => {
      const el = keepOutRef.current;
      if (!el) { zoneL = zoneR = 0; return; }
      const c = canvas.getBoundingClientRect();
      const z = el.getBoundingClientRect();
      zoneL = z.left - c.left - KEEP_OUT_PAD;
      zoneR = z.right - c.left + KEEP_OUT_PAD;
    };

    /** Horizontal bounds of a gutter, as [min, max]. */
    const bounds = (side: -1 | 1): [number, number] =>
      side < 0 ? [0, Math.max(0, zoneL)] : [Math.min(w, zoneR), w];

    const spawn = (side: -1 | 1): Particle => {
      const [lo, hi] = bounds(side);
      const a = Math.random() * Math.PI * 2;
      return {
        x: lo + Math.random() * Math.max(1, hi - lo),
        y: Math.random() * h,
        // Start moving, so nothing has to accelerate from a dead stop.
        vx: Math.cos(a) * MAX_SPEED,
        vy: Math.sin(a) * MAX_SPEED,
        rot: (Math.random() - 0.5) * 0.6,
        spin: 0,
        size: MIN_SIZE + Math.random() * (MAX_SIZE - MIN_SIZE),
        kind: Math.floor(Math.random() * KINDS),
        wander: Math.random() * Math.PI * 2,
        side,
      };
    };

    const resize = () => {
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      measureZone();
      const nLeft = Math.round((Math.max(0, zoneL) * h) / DENSITY);
      const nRight = Math.round((Math.max(0, w - zoneR) * h) / DENSITY);
      particles = [
        ...Array.from({ length: nLeft }, () => spawn(-1)),
        ...Array.from({ length: nRight }, () => spawn(1)),
      ];
    };

    /**
     * Apply a steering force toward `desired`, Reynolds-style:
     * steer = desired - velocity, capped at MAX_FORCE.
     */
    const steerTo = (p: Particle, dx: number, dy: number, weight = 1) => {
      const len = Math.hypot(dx, dy);
      if (len < 0.0001) return;
      // Desired velocity: full speed in that direction.
      const desX = (dx / len) * MAX_SPEED;
      const desY = (dy / len) * MAX_SPEED;
      let sx = desX - p.vx;
      let sy = desY - p.vy;
      const slen = Math.hypot(sx, sy);
      if (slen > MAX_FORCE) {
        sx = (sx / slen) * MAX_FORCE;
        sy = (sy / slen) * MAX_FORCE;
      }
      p.vx += sx * weight;
      p.vy += sy * weight;
    };

    const step = () => {
      for (const p of particles) {
        const [lo, hi] = bounds(p.side);

        // --- wander: steer toward a target riding a projected circle ---
        p.wander += (Math.random() - 0.5) * WANDER_JITTER;
        const speed = Math.hypot(p.vx, p.vy) || 1;
        // Circle centre sits ahead of the particle, along its heading.
        const cx = p.x + (p.vx / speed) * WANDER_DIST;
        const cy = p.y + (p.vy / speed) * WANDER_DIST;
        const tx = cx + Math.cos(p.wander) * WANDER_RADIUS;
        const ty = cy + Math.sin(p.wander) * WANDER_RADIUS;
        steerTo(p, tx - p.x, ty - p.y);

        // --- stay within walls: steer inward when near a boundary ---
        // Reynolds' "stay within walls": rather than bouncing, the particle
        // is asked to head inward, and the usual turn limit makes that a
        // smooth arc.
        if (p.x < lo + EDGE_MARGIN) steerTo(p, 1, 0, 1.6);
        else if (p.x > hi - EDGE_MARGIN) steerTo(p, -1, 0, 1.6);
        if (p.y < EDGE_MARGIN) steerTo(p, 0, 1, 1.6);
        else if (p.y > h - EDGE_MARGIN) steerTo(p, 0, -1, 1.6);

        // --- cursor repulsion (an impulse, not a steering force) ---
        if (cursor) {
          const ddx = p.x - cursor.x, ddy = p.y - cursor.y;
          const dist = Math.hypot(ddx, ddy);
          if (dist < REPEL_RADIUS && dist > 0.001) {
            const f = (1 - dist / REPEL_RADIUS) * REPEL_FORCE;
            p.vx += (ddx / dist) * f;
            p.vy += (ddy / dist) * f;
            p.spin += (ddx * p.vy - ddy * p.vx) * 0.0004;
          }
        }

        // Bleed off burst velocity, but never below the cruising speed:
        // friction alone would grind the field to a halt.
        const sp = Math.hypot(p.vx, p.vy);
        if (sp > MAX_SPEED) {
          const damped = Math.max(MAX_SPEED, Math.min(MAX_BURST, sp * FRICTION));
          p.vx = (p.vx / sp) * damped;
          p.vy = (p.vy / sp) * damped;
        }

        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.spin;
        p.spin *= FRICTION;

        // Hard backstop: only a violent cursor shove should ever reach this.
        const m = p.size * 0.5;
        if (p.x < lo - m) { p.x = lo - m; p.vx = Math.abs(p.vx); }
        else if (p.x > hi + m) { p.x = hi + m; p.vx = -Math.abs(p.vx); }
        if (p.y < -m) { p.y = -m; p.vy = Math.abs(p.vy); }
        else if (p.y > h + m) { p.y = h + m; p.vy = -Math.abs(p.vy); }
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = dark.matches ? "rgba(148,163,184,0.22)" : "rgba(100,116,139,0.18)";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 1;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const p of particles) {
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        drawIcon(ctx, p.kind, p.size);
        ctx.restore();
      }
    };

    const frame = () => {
      if (!reducedMotion) step();
      draw();
      rafId = requestAnimationFrame(frame);
    };

    const toCanvas = (clientX: number, clientY: number) => {
      const r = canvas.getBoundingClientRect();
      return { x: clientX - r.left, y: clientY - r.top };
    };
    const onMove = (e: MouseEvent) => { cursor = toCanvas(e.clientX, e.clientY); };
    const onLeave = () => { cursor = null; };
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      cursor = t ? toCanvas(t.clientX, t.clientY) : null;
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    // Listen on window: the canvas is pointer-events-none so it never
    // receives events itself, but we still want to react to the cursor.
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseleave", onLeave);
    window.addEventListener("touchmove", onTouch, { passive: true });
    window.addEventListener("touchend", onLeave);
    dark.addEventListener("change", draw);
    rafId = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("touchend", onLeave);
      dark.removeEventListener("change", draw);
    };
  }, [keepOutRef]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 h-full w-full ${className ?? ""}`}
    />
  );
}

/** Draw one glyph centred on the origin. `s` is its bounding size in px. */
function drawIcon(ctx: CanvasRenderingContext2D, kind: number, s: number) {
  const h = s / 2;
  switch (kind) {
    case 0: {
      const b = h * 0.6;
      ctx.strokeRect(-b, -b, b * 2, b * 2);
      ctx.beginPath();
      for (let i = -1; i <= 1; i++) {
        const o = i * b * 0.6;
        ctx.moveTo(o, -b); ctx.lineTo(o, -h);
        ctx.moveTo(o, b); ctx.lineTo(o, h);
        ctx.moveTo(-b, o); ctx.lineTo(-h, o);
        ctx.moveTo(b, o); ctx.lineTo(h, o);
      }
      ctx.stroke();
      return;
    }
    case 1: {
      const r = Math.max(1.2, s * 0.09);
      const pts: [number, number][] = [[-h * 0.7, -h * 0.55], [-h * 0.7, h * 0.55], [h * 0.7, 0]];
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.moveTo(pts[1][0], pts[1][1]); ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.stroke();
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
      return;
    }
    case 2: {
      const t = h * 0.22;
      ctx.beginPath();
      ctx.moveTo(0, -h); ctx.lineTo(t, -t); ctx.lineTo(h, 0); ctx.lineTo(t, t);
      ctx.lineTo(0, h); ctx.lineTo(-t, t); ctx.lineTo(-h, 0); ctx.lineTo(-t, -t);
      ctx.closePath(); ctx.stroke();
      return;
    }
    case 3: {
      const a = h * 0.75, k = h * 0.45;
      ctx.beginPath();
      ctx.moveTo(-k, -a * 0.6); ctx.lineTo(-a, 0); ctx.lineTo(-k, a * 0.6);
      ctx.moveTo(k, -a * 0.6); ctx.lineTo(a, 0); ctx.lineTo(k, a * 0.6);
      ctx.moveTo(-h * 0.2, a * 0.7); ctx.lineTo(h * 0.2, -a * 0.7);
      ctx.stroke();
      return;
    }
    case 4: {
      const bw = h, bh = h * 0.7, r = h * 0.3;
      ctx.beginPath();
      ctx.moveTo(-bw + r, -bh);
      ctx.arcTo(bw, -bh, bw, bh, r);
      ctx.arcTo(bw, bh, -bw, bh, r);
      ctx.lineTo(-bw * 0.3, bh);
      ctx.lineTo(-bw * 0.6, bh + h * 0.45);
      ctx.lineTo(-bw * 0.65, bh);
      ctx.arcTo(-bw, bh, -bw, -bh, r);
      ctx.arcTo(-bw, -bh, bw, -bh, r);
      ctx.closePath(); ctx.stroke();
      const dr = Math.max(0.9, s * 0.05);
      for (const dx of [-0.45, 0, 0.45]) { ctx.beginPath(); ctx.arc(dx * bw, 0, dr, 0, Math.PI * 2); ctx.fill(); }
      return;
    }
    case 5: {
      const ro = h, ri = h * 0.7, teeth = 8;
      ctx.beginPath();
      for (let i = 0; i < teeth * 2; i++) {
        const rad = i % 2 === 0 ? ro : ri;
        const ang = (i / (teeth * 2)) * Math.PI * 2;
        const x = Math.cos(ang) * rad, y = Math.sin(ang) * rad;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath(); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, h * 0.3, 0, Math.PI * 2); ctx.stroke();
      return;
    }
    default: {
      const label = LABELS[kind - 6] ?? "AI";
      ctx.font = `${Math.max(8, s * 0.75)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 0, 0);
    }
  }
}
