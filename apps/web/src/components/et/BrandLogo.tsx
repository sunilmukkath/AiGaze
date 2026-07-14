import Image from "next/image";

interface BrandLogoProps {
  /** Height of the logo lockup in px */
  height?: number;
  className?: string;
  priority?: boolean;
  /** Optional text next to logo — usually omit; aigaze-logo is a full lockup */
  withWordmark?: boolean;
  wordmarkClassName?: string;
}

/** AI Gaze product mark (full lockup). Elastic Tree company brand lives in the footer. */
export function BrandLogo({
  height = 36,
  className = "",
  priority = false,
  withWordmark = false,
  wordmarkClassName = "",
}: BrandLogoProps) {
  const width = Math.round(height * (1536 / 1024));
  return (
    <span className={`inline-flex items-center gap-2.5 shrink-0 ${className}`}>
      <Image
        src="/aigaze-logo.png"
        alt={withWordmark ? "" : "AI Gaze"}
        width={width}
        height={height}
        className="shrink-0 object-contain drop-shadow-[0_0_12px_rgba(232,168,32,0.25)]"
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
    </span>
  );
}
