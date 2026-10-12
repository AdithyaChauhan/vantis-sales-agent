import { useId } from "react";

/**
 * The Vantis mark: a sun rising over the horizon (public/favicon.svg uses the same shapes).
 * Decorative: the company name is always next to it, so screen readers skip it.
 */

// Seven rays around the sun's centre (32, 31), from radius 14.5 to 19.5, every 22.5 degrees.
const RAYS = [
  [45.4, 25.45, 50.02, 23.54],
  [42.25, 20.75, 45.79, 17.21],
  [37.55, 17.6, 39.46, 12.98],
  [32, 16.5, 32, 11.5],
  [26.45, 17.6, 24.54, 12.98],
  [21.75, 20.75, 18.21, 17.21],
  [18.6, 25.45, 13.98, 23.54],
];

/** A gradient id that is unique on the page and safe inside url(#...). */
function useGradientId() {
  return `sun-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

function Sun({ gradientId }) {
  return (
    <>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffc63f" />
          <stop offset="1" stopColor="#ff7a12" />
        </linearGradient>
      </defs>
      <g stroke="#ffa12b" strokeWidth="2.6" strokeLinecap="round">
        {RAYS.map(([x1, y1, x2, y2]) => (
          <line key={`${x1}-${y1}`} x1={x1} y1={y1} x2={x2} y2={y2} />
        ))}
      </g>
      <path d="M21 31 A11 11 0 0 1 43 31 Z" fill={`url(#${gradientId})`} />
    </>
  );
}

/** The full logo: the sun with a navy horizon swoosh under it (the swoosh follows the text colour). */
export default function Logo({ className }) {
  const gradientId = useGradientId();
  return (
    <svg className={className} viewBox="2 6 60 34" aria-hidden="true" focusable="false">
      <Sun gradientId={gradientId} />
      <path
        className="logo-horizon"
        d="M6 35 C 20 30.5, 44 30.5, 58 35"
        fill="none"
        stroke="currentColor"
        strokeWidth="3.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** The sun alone over a short orange horizon line, for the assistant's avatar. */
export function SunMark({ className }) {
  const gradientId = useGradientId();
  return (
    <svg className={className} viewBox="10 8 44 28" aria-hidden="true" focusable="false">
      <Sun gradientId={gradientId} />
      <path d="M16 33.5 H48" stroke="#ff8a1f" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}
