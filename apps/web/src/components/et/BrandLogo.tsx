import Image from "next/image";
import { ElasticTreeLogo } from "./ElasticTreeLogo";

interface BrandLogoProps {
  /** Height of the AI Gaze logo lockup in px */
  height?: number;
  className?: string;
  priority?: boolean;
  /** Show Elastic Tree mark beside AI Gaze (default true). */
  showElasticTree?: boolean;
  /** Optional text next to logo — usually omit; aigaze-logo is a full lockup */
  withWordmark?: boolean;
  wordmarkClassName?: string;
}

/** AI Gaze product mark with optional Elastic Tree company logo. */
export function BrandLogo({
  height = 48,
  className = "",
  priority = false,
  showElasticTree = true,
  withWordmark = false,
  wordmarkClassName = "",
}: BrandLogoProps) {
  const width = Math.round(height * (1419 / 488));
  const etHeight = Math.max(18, Math.round(height * 0.58));
  return (
    <span className={`inline-flex items-center gap-3 shrink-0 ${className}`}>
      <Image
        src="/aigaze-logo.png"
        alt={withWordmark ? "" : "AI Gaze"}
        width={width}
        height={height}
        className="shrink-0 object-contain"
        priority={priority}
        style={{ height, width: "auto", maxWidth: width }}
      />
      {withWordmark && (
        <span
          className={`et-brand-name font-bold tracking-tight font-[family-name:var(--font-outfit)] ${wordmarkClassName}`}
        >
          AI Gaze
        </span>
      )}
      {showElasticTree && (
        <>
          <span
            aria-hidden
            className="hidden sm:block w-px self-stretch min-h-[1.5rem] bg-white/20 shrink-0"
          />
          <ElasticTreeLogo height={etHeight} priority={priority} className="hidden sm:block opacity-95" />
        </>
      )}
    </span>
  );
}
